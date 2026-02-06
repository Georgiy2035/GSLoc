#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class SplitParams:
    timestamp_to_find: int = 1846952165000
    window_frames: int = 5
    db_distance_threshold_m: float = 0.5
    db_angle_threshold_deg: float = 30.0
    query_neighbor_radius_m: float = 1.0
    lidar_col: str = "lidar_joined_ts"


def read_frames_csv(frames_csv: Path) -> pd.DataFrame:
    if not frames_csv.exists():
        raise FileNotFoundError(f"Frames CSV not found: {frames_csv}")

    df = pd.read_csv(frames_csv)
    # Normalize column names: strip whitespace and leading '#'
    df.columns = [str(c).strip().lstrip('#').strip() for c in df.columns]

    # Support two formats:
    #  1) Original frames.csv with columns: pose_ts, x, y, z, qx, qy, qz, qw, lidar_joined_ts, ...
    #  2) New poses.csv with columns: ts, px, py, pz, qx, qy, qz, qw

    cols = set(df.columns)

    # If new poses.csv format, normalize to the original schema used by this script
    if {"ts", "px", "py", "pz", "qx", "qy", "qz", "qw"}.issubset(cols):
        # Rename pose and position columns
        df = df.rename(columns={
            "ts": "pose_ts",
            "px": "x",
            "py": "y",
            "pz": "z",
        })
        # Ensure integer pose timestamps (round if needed)
        df["pose_ts"] = df["pose_ts"].round().astype("int64")
        # If no LiDAR timestamp exists, mirror from pose_ts for downstream expectations
        if "lidar_joined_ts" not in df.columns:
            df["lidar_joined_ts"] = df["pose_ts"].astype("int64")

    # Validate required columns after normalization
    expected_pose_col = "pose_ts"
    if expected_pose_col not in df.columns:
        raise KeyError(
            f"Column '{expected_pose_col}' not found in {frames_csv}. Columns: {list(df.columns)}"
        )
    df[expected_pose_col] = df[expected_pose_col].round().astype("int64")

    for col in ("x", "y", "z", "qx", "qy", "qz", "qw"):
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in frames CSV: {frames_csv}")

    return df


def find_split_index(pose_ts: np.ndarray, target_ts: int) -> Tuple[int, int]:
    """Return (split_index, matched_ts) where split_index is index of pose_ts closest to target_ts."""
    idx = int(np.argmin(np.abs(pose_ts - int(target_ts))))
    return idx, int(pose_ts[idx])


def split_dataframe(df: pd.DataFrame, split_index: int, window_frames: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    start_database = 0
    end_database = max(0, split_index - window_frames)  # exclusive in iloc slicing

    start_query = min(len(df), split_index + window_frames)
    end_query = len(df)

    database_df = df.iloc[start_database:end_database].reset_index(drop=True)
    query_df = df.iloc[start_query:end_query].reset_index(drop=True)

    return database_df, query_df


def filter_database_by_distance_and_angle(df: pd.DataFrame, dist_thr_m: float, angle_thr_deg: float) -> pd.DataFrame:
    if df.empty:
        return df.copy()

    poses = df[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy()

    angle_thr_rad = np.deg2rad(angle_thr_deg)

    kept_indices: list[int] = [0]
    for i in range(1, len(poses)):
        prev_pose = poses[kept_indices[-1]]
        curr_pose = poses[i]

        # Euclidean distance between positions
        distance = np.linalg.norm(curr_pose[:3] - prev_pose[:3])

        # Angular distance using quaternion dot product
        dot = float(np.dot(curr_pose[3:], prev_pose[3:]))
        dot = float(np.clip(dot, -1.0, 1.0))
        angle_rad = 2.0 * np.arccos(dot)

        if (distance > dist_thr_m) or (angle_rad > angle_thr_rad):
            kept_indices.append(i)

    return df.iloc[kept_indices].reset_index(drop=True)


def filter_query_by_db_neighbors(query_df: pd.DataFrame, db_df: pd.DataFrame, radius_m: float) -> pd.DataFrame:
    if query_df.empty or db_df.empty:
        return query_df.copy()

    db_xyz = db_df[["x", "y", "z"]].to_numpy(dtype=np.float64)
    q_xyz = query_df[["x", "y", "z"]].to_numpy(dtype=np.float64)

    # Compute pairwise distances (Q x D). For moderate sizes this is fine and mirrors the notebook.
    # dists[i, j] = ||q_i - db_j||
    diffs = q_xyz[:, None, :] - db_xyz[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    valid_mask = (dists < float(radius_m)).any(axis=1)

    return query_df.loc[valid_mask].reset_index(drop=True)


def ensure_required_modalities(df: pd.DataFrame, lidar_col: str) -> pd.DataFrame:
    if lidar_col not in df.columns:
        available = ", ".join(list(df.columns))
        raise KeyError(
            f"Required LiDAR timestamp column '{lidar_col}' not found. Available columns: {available}"
        )
    # Drop any rows without lidar timestamp (should not happen if frames were built with dropping NAs)
    out = df.dropna(subset=[lidar_col]).copy()
    # Enforce integer dtype now that NAs are gone
    out[lidar_col] = out[lidar_col].astype("int64")
    return out


def rename_columns_for_output(df: pd.DataFrame, lidar_col: str) -> pd.DataFrame:
    # Preserve original column names entirely (including 'pose_ts' and lidar column).
    return df




def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Split frames.csv into database and query based on a target timestamp (±window), "
            "filter database by distance/angle, and keep only query frames with a database "
            "neighbor within a radius. Outputs are db_lidar_frames.csv and filtered_query_lidar_frames.csv."
        )
    )
    parser.add_argument(
        "--mav0-dir",
        type=Path,
        required=True,
        help="Path to the dataset mav0 directory (contains frames.csv, lidar*/cam*/depth folders)",
    )
    parser.add_argument(
        "--frames-csv",
        type=Path,
        default=None,
        help="Path to frames.csv (default: <mav0-dir>/frames.csv)",
    )
    parser.add_argument(
        "--timestamp-to-find",
        type=int,
        default=SplitParams.timestamp_to_find,
        help="Timestamp (ns) to split around; nearest pose_ts is used",
    )
    parser.add_argument(
        "--window-frames",
        type=int,
        default=SplitParams.window_frames,
        help="Half-gap (in frames) around the split index to exclude from both sides",
    )
    parser.add_argument(
        "--db-distance-m",
        type=float,
        default=SplitParams.db_distance_threshold_m,
        help="Database filtering: keep a frame if distance from last kept > this threshold (meters)",
    )
    parser.add_argument(
        "--db-angle-deg",
        type=float,
        default=SplitParams.db_angle_threshold_deg,
        help="Database filtering: keep a frame if angle from last kept > this threshold (degrees)",
    )
    parser.add_argument(
        "--query-neighbor-radius-m",
        type=float,
        default=SplitParams.query_neighbor_radius_m,
        help="Keep only query frames that have a database neighbor within this radius (meters)",
    )
    parser.add_argument(
        "--lidar-col",
        type=str,
        default=SplitParams.lidar_col,
        help="Column name of LiDAR timestamps in frames.csv (default: 'lidar_joined_ts')",
    )
    parser.add_argument(
        "--db-output",
        type=Path,
        default=None,
        help="Output CSV for database (default: <mav0-dir>/db_lidar_frames.csv)",
    )
    parser.add_argument(
        "--query-output",
        type=Path,
        default=None,
        help="Output CSV for filtered query (default: <mav0-dir>/filtered_query_lidar_frames.csv)",
    )

    args = parser.parse_args()

    mav0_dir: Path = args.mav0_dir.resolve()
    frames_csv: Path = (args.frames_csv.resolve() if args.frames_csv is not None else (mav0_dir / "frames.csv"))
    # Auto-detect poses.csv if frames.csv is not present
    if not frames_csv.exists():
        poses_csv_fallback = mav0_dir / "poses.csv"
        if poses_csv_fallback.exists():
            frames_csv = poses_csv_fallback

    # 1) Load frames
    frames_df = read_frames_csv(frames_csv)

    # 2) Ensure LiDAR modality is present and usable
    frames_df = ensure_required_modalities(frames_df, lidar_col=args.lidar_col)

    # 4) Split at nearest timestamp with ±window
    pose_ts = frames_df["pose_ts"].to_numpy(dtype=np.int64)
    split_idx, matched_ts = find_split_index(pose_ts, args.timestamp_to_find)
    print(f"Split at index {split_idx} (pose_ts={matched_ts}), target={args.timestamp_to_find}")

    db_full, query_full = split_dataframe(frames_df, split_idx, int(args.window_frames))
    print(f"Initial database frames: {len(db_full)}, query frames: {len(query_full)}")

    # 5) Filter database by distance and angle
    db_filtered = filter_database_by_distance_and_angle(
        db_full,
        dist_thr_m=float(args.db_distance_m),
        angle_thr_deg=float(args.db_angle_deg),
    )
    print(f"Database after distance/angle filtering: {len(db_filtered)} frames")

    # 6) Keep only queries with DB neighbor within radius
    query_filtered = filter_query_by_db_neighbors(
        query_full,
        db_filtered,
        radius_m=float(args.query_neighbor_radius_m),
    )
    print(f"Query after neighbor filtering: {len(query_filtered)} frames")

    # 7) Rename columns for downstream usage and save
    db_out = rename_columns_for_output(db_filtered, lidar_col=args.lidar_col)
    query_out = rename_columns_for_output(query_filtered, lidar_col=args.lidar_col)

    db_out_path = (args.db_output.resolve() if args.db_output is not None else (mav0_dir / "db_lidar_frames.csv"))
    query_out_path = (args.query_output.resolve() if args.query_output is not None else (mav0_dir / "filtered_query_lidar_frames.csv"))

    db_out_path.parent.mkdir(parents=True, exist_ok=True)
    query_out_path.parent.mkdir(parents=True, exist_ok=True)

    db_out.to_csv(db_out_path, index=False)
    query_out.to_csv(query_out_path, index=False)

    print(f"Saved database to {db_out_path}")
    print(f"Saved filtered query to {query_out_path}")


if __name__ == "__main__":
    main()
