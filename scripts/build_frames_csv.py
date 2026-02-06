#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd


SUPPORTED_MODALITIES_DEFAULT = (
    "lidar0",
    "lidar1",
    "lidar_joined",
    "cam0",
    "cam1",
    "cam2",
    "cam3",
    "depth",
)


@dataclass
class ModalitySpec:
    name: str
    data_dir: Path
    exts: tuple[str, ...]


def list_files_with_exts(directory: Path, exts: tuple[str, ...]) -> List[Path]:
    files: List[Path] = []
    if not directory.exists():
        return files
    for ext in exts:
        files.extend(sorted(directory.glob(f"*.{ext}")))
    return files


def list_timestamps_from_data_dir(mod_dir: Path) -> np.ndarray:
    """
    List timestamps for a modality by parsing filenames in `<mod_dir>/data/`.

    Assumes filenames are `<timestamp>.<ext>` where timestamp is an integer in ns.
    Tries reasonable extensions depending on directory name.
    """
    # Heuristics: lidar -> .pcd, otherwise try common image/depth exts
    if mod_dir.name.startswith("lidar"):
        exts = ("pcd",)
        # For lidar0/lidar1 files lie directly in the modality directory
        search_dir = mod_dir if mod_dir.name in ("lidar0", "lidar1") else (mod_dir / "data")
    else:
        exts = ("png", "jpg", "jpeg", "tiff", "bmp", "npz", "npy")
        search_dir = mod_dir / "data"

    files = list_files_with_exts(search_dir, exts)
    if not files:
        # Fallback: if there's a data.csv with timestamps, try reading first col
        csv_path = mod_dir / "data.csv"
        if csv_path.exists():
            try:
                df = pd.read_csv(csv_path)
                # Try common column names
                for col in ("timestamp", "ts", "time", "frame_timestamp"):
                    if col in df.columns:
                        ts = (
                            df[col]
                            .astype("int64", errors="ignore")
                            .astype("int64")
                            .to_numpy()
                        )
                        ts.sort()
                        return ts
                # Else assume first column
                first_col = df.columns[0]
                ts = df[first_col].astype("int64").to_numpy()
                ts.sort()
                return ts
            except Exception:
                pass

    timestamps: List[int] = []
    for f in files:
        try:
            timestamps.append(int(f.stem))
        except ValueError:
            continue
    ts_arr = np.array(sorted(timestamps), dtype=np.int64)
    return ts_arr


def assign_nearest_unique(ref_ts: np.ndarray, mod_ts: np.ndarray, max_dt_ns: Optional[int]) -> List[Optional[int]]:
    """
    For each ref timestamp, assign the nearest unique modality timestamp (no reuse),
    respecting an optional threshold `max_dt_ns`. If no candidate within threshold, return None.

    Greedy, single-pass, monotonic assignment over sorted arrays.
    """
    if mod_ts.size == 0:
        return [None] * int(ref_ts.size)

    out: List[Optional[int]] = []
    k = 0  # pointer into mod_ts, always non-decreasing
    n = int(mod_ts.size)

    for t in ref_ts:
        if k >= n:
            out.append(None)
            continue
        # Advance while the next is closer to current ref t
        while k + 1 < n and abs(int(mod_ts[k + 1]) - int(t)) <= abs(int(mod_ts[k]) - int(t)):
            k += 1
        candidate = int(mod_ts[k])
        if (max_dt_ns is not None) and (abs(candidate - int(t)) > max_dt_ns):
            out.append(None)
        else:
            out.append(candidate)
            k += 1  # consume unique
    return out


def build_frames(
    mav0_dir: Path,
    max_dt_ms: int,
    modalities: Iterable[str] = SUPPORTED_MODALITIES_DEFAULT,
) -> pd.DataFrame:
    # 1) Reference timestamps from traj.txt (ns)
    traj_path = mav0_dir / "traj.txt"
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing traj.txt at {traj_path}")
    traj_df = pd.read_table(
        traj_path,
        header=None,
        sep=" ",
        names=["pose_ts", "x", "y", "z", "qx", "qy", "qz", "qw"],
        dtype={"pose_ts": np.float64},  # some tools save as float sci-notation
    )
    # Harden timestamps to int64 ns
    ref_ts = traj_df["pose_ts"].round().astype("int64").to_numpy()

    # 2) Collect modality timestamps
    modality_to_ts: dict[str, np.ndarray] = {}
    for name in modalities:
        mod_dir = mav0_dir / name
        if not mod_dir.exists():
            continue
        ts = list_timestamps_from_data_dir(mod_dir)
        if ts.size:
            modality_to_ts[name] = ts
        else:
            # Warn: modality has no timestamps discovered
            print(f"[WARN] No samples found for modality '{name}' in {mod_dir}")

    # 3) Assign nearest unique per modality
    max_dt_ns = int(max_dt_ms * 1_000_000)
    data = {
        "pose_ts": ref_ts,
        "x": traj_df["x"].to_numpy(),
        "y": traj_df["y"].to_numpy(),
        "z": traj_df["z"].to_numpy(),
        "qx": traj_df["qx"].to_numpy(),
        "qy": traj_df["qy"].to_numpy(),
        "qz": traj_df["qz"].to_numpy(),
        "qw": traj_df["qw"].to_numpy(),
    }
    for name, ts in modality_to_ts.items():
        assigned = assign_nearest_unique(ref_ts, ts, max_dt_ns)
        # Use pandas nullable Int64 to allow missing values
        data[f"{name}_ts"] = pd.Series(assigned, dtype="Int64")

    frames_df = pd.DataFrame(data)

    # Drop rows where not all modalities (that were discovered) are present
    modality_cols = [f"{name}_ts" for name in modality_to_ts.keys()]
    if modality_cols:
        before = len(frames_df)
        frames_df = frames_df.dropna(subset=modality_cols)
        after = len(frames_df)
        print(f"Dropped {before - after} rows without all selected modalities present (threshold {max_dt_ms} ms)")
    return frames_df


def main() -> None:
    parser = argparse.ArgumentParser(description="Build frames.csv by time-syncing modalities to traj.txt (nearest unique within threshold)")
    parser.add_argument(
        "--mav0-dir",
        type=Path,
        required=True,
        help="Path to the dataset mav0 directory (contains traj.txt, lidar*/cam*/depth folders)",
    )
    parser.add_argument(
        "--max-dt-ms",
        type=int,
        default=30,
        help="Maximum allowed time difference (ms) between pose_ts and a modality sample to be assigned",
    )
    parser.add_argument(
        "--modalities",
        type=str,
        default=",".join(SUPPORTED_MODALITIES_DEFAULT),
        help="Comma-separated modality folder names to include if present",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output CSV path (default: <mav0-dir>/frames.csv)",
    )
    args = parser.parse_args()

    mav0_dir: Path = args.mav0_dir.resolve()
    modalities = tuple([s.strip() for s in args.modalities.split(",") if s.strip()])

    frames_df = build_frames(mav0_dir=mav0_dir, max_dt_ms=args.max_dt_ms, modalities=modalities)

    out_path = args.output if args.output is not None else (mav0_dir / "frames.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    frames_df.to_csv(out_path, index=False)
    print(f"Saved frames to {out_path}")


if __name__ == "__main__":
    main()


