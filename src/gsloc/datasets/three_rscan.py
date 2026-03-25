"""3RScan dataset loader and metadata builder.

This module provides:
- `build_3rscan_meta_parquet`: scans the dataset, loads per-frame poses, and writes a single `meta.parquet`.
- `ThreeRScan`: a PyTorch `Dataset` that loads RGB images from all scenes using the `meta.parquet`.

Dataset layout assumed (matches your dataset under `/mnt/external_usb_hdd/6YL/Datasets/3RScan`):

`scenes/<scene_id>/sequence/frame-XXXXXX.color.jpg`
`scenes/<scene_id>/sequence/frame-XXXXXX.pose.txt`  (4x4 matrix, one row per line)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.spatial.transform import Rotation as R
from torch import Tensor
from torch.utils.data import Dataset

from opr.datasets.augmentations import DefaultImageTransform
from mmpr.modules.vis_utils import quaternion_angle


def _read_pose_matrix(pose_path: Path) -> np.ndarray:
    """Read a 4x4 pose matrix from a 3RScan `*.pose.txt` file."""
    mat = np.loadtxt(str(pose_path), dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected pose matrix (4,4), got {mat.shape} for {pose_path}")
    return mat


def _matrix_to_pose7(T_wc: np.ndarray) -> List[float]:
    """Convert 4x4 world-from-camera matrix to [tx, ty, tz, qx, qy, qz, qw]."""
    t = T_wc[:3, 3].astype(np.float64)
    q_xyzw = R.from_matrix(T_wc[:3, :3]).as_quat().astype(np.float64)  # x,y,z,w
    return [
        float(t[0]),
        float(t[1]),
        float(t[2]),
        float(q_xyzw[0]),
        float(q_xyzw[1]),
        float(q_xyzw[2]),
        float(q_xyzw[3]),
    ]


def iter_3rscan_frames(dataset_root: Union[str, Path]) -> Iterable[Tuple[str, Path, Path]]:
    """Yield (scene_id, image_path, pose_path) for all frames that have a pose."""
    root = Path(dataset_root)
    scenes_root = root / "scenes"
    if not scenes_root.exists():
        raise FileNotFoundError(f"Expected scenes folder at {scenes_root}")

    for scene_dir in sorted(p for p in scenes_root.iterdir() if p.is_dir()):
        scene_id = scene_dir.name
        seq_dir = scene_dir / "sequence"
        if not seq_dir.exists():
            continue
        for img_path in sorted(seq_dir.glob("*.color.jpg")):
            pose_path = img_path.with_suffix("").with_suffix(".pose.txt")  # frame-XXXXXX.pose.txt
            if pose_path.exists():
                yield scene_id, img_path, pose_path


def build_3rscan_df(
        dataset_root: Union[str, Path],
        *,
        limit: Optional[int] = None,
        log_every: int = 50_000,
    ) -> Path:
        """Build a single `meta.parquet` for 3RScan.

        The resulting file contains:
        - `idx` (int): unique row id
        - `scene` (str): scene id (folder name under `scenes/`)
        - `pose` (list[float]): [tx, ty, tz, qx, qy, qz, qw] in world coordinates
        - `image_path` (str): absolute path to RGB frame (`*.color.jpg`)
        """
        root = Path(dataset_root)

        rows: List[Dict[str, Any]] = []
        n = 0

        logger.info("Scanning 3rscan dataset...")

        for scene_id, img_path, pose_path in iter_3rscan_frames(root):
            try:
                T_wc = _read_pose_matrix(pose_path)
                pose7 = _matrix_to_pose7(T_wc)
            except Exception:
                logger.exception(f"Failed reading pose for {pose_path}")
                continue

            rows.append(
                {
                    "idx": n,
                    "scene": scene_id,
                    "pose": pose7,
                    "image_path": str(img_path),
                }
            )
            n += 1
            if log_every and (n % log_every == 0):
                logger.info(f"Scanned {n:,} frames...")
            if limit is not None and n >= limit:
                break

        if not rows:
            raise RuntimeError(f"No frames with poses found under {root}/scenes/*/sequence")

        df = pd.DataFrame(rows)
        logger.info(f"Scanned {n} rows")

        return df
        

class ThreeRScan(Dataset):
    """3RScan dataset that loads RGB frames from all scenes."""

    _scene_to_room_map: Optional[Dict[str, str]] = None

    def __init__(
        self,
        dataset_root: Union[str, Path] = "/mnt/external_usb_hdd/6YL/Datasets/3RScan",
        *,
        meta_path: Optional[Union[str, Path]] = None, 
        meta_file: str = "meta.parquet",
        rebuild_meta: bool = False,
        save_meta: bool = False,
        limit: Optional[int] = None,
        image_transform: Any = DefaultImageTransform(resize=(320, 192), train=False),
    ) -> None:
    
        super().__init__()
        self.dataset_root = Path(dataset_root)
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Given dataset_root={self.dataset_root} doesn't exist")

        self.meta_path = self.dataset_root / meta_file if meta_path is None else Path(meta_path) / meta_file  
        
        if self.meta_path.exists() and not rebuild_meta:
            self.df = pd.read_parquet(self.meta_path)
        else:
            logger.info("Rebuilding metadata for 3rscan dataset" if self.meta_path.exists() else "Metadata didn't found, rebuilding metadata for 3rscan dataset")
            self.df = build_3rscan_df(self.dataset_root, limit=limit)
        
        if limit is not None:
            self.df = self.df.iloc[:limit].reset_index(drop=True)

        # missing cols checking
        missing_cols = {"idx", "pose", "image_path"} - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"{self.meta_path} is missing columns: {sorted(missing_cols)}")
        if "scene" not in self.df.columns:
            self.df["scene"] = self.df["image_path"].map(lambda p: Path(p).parents[1].name)

        self.image_transform = image_transform

        if save_meta:
            self.save_meta_parquet(self.meta_path, meta_file)

    def __len__(self) -> int:  
        return int(len(self.df))

    def _load_image(self, image_path: Union[str, Path]) -> Tensor:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_transform is not None:
            img = self.image_transform(img)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.ascontiguousarray(img))
            if img.ndim == 3:
                img = img.permute(2, 0, 1)  # HWC -> CHW
            img = img.float() / 255.0
        return img

    def __getitem__(self, idx: int) -> Dict[str, Any]:  
        row = self.df.iloc[int(idx)]
        image_path = row["image_path"]
        scene = row["scene"]
        pose = torch.tensor(np.asarray(row["pose"], dtype=np.float32), dtype=torch.float32)
        image = self._load_image(image_path)
        return {
            "idx": torch.tensor(int(row["idx"]), dtype=torch.int64),
            # Keep the original scene id so callers can reason about rooms/scenes.
            "scene_id": str(scene),
            "scene": torch.tensor(hash(str(scene))),
            "pose": pose,
            "image_main": image,
        }

    @classmethod
    def _get_scene_to_room_map(
        cls,
        *,
        room_json_path: Union[str, Path] = "/mnt/external_usb_hdd/6YL/Datasets/3RScan/files/3RScan.json",
    ) -> Dict[str, str]:
        """
        Build mapping `scene_id -> room_reference`.

        In `3RScan.json`, scenes that belong to the same physical room are grouped under the same
        top-level object, with the top-level `"reference"` and all entries from `"scans"[]."reference"`.
        """
        if cls._scene_to_room_map is not None:
            return cls._scene_to_room_map

        room_json_path = Path(room_json_path)
        if not room_json_path.exists():
            raise FileNotFoundError(f"Missing 3RScan room file: {room_json_path}")

        with room_json_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON list at top-level in {room_json_path}")

        scene_to_room: Dict[str, str] = {}
        for entry in data:
            if not isinstance(entry, dict):
                continue
            room_ref = entry.get("reference")
            scans = entry.get("scans")
            if room_ref is None or not isinstance(scans, list):
                continue

            # The room contains the top-level reference scan and all rescan references in "scans".
            for scan_entry in [{"reference": room_ref}] + scans:
                if not isinstance(scan_entry, dict):
                    continue
                scan_ref = scan_entry.get("reference")
                if scan_ref is None:
                    continue
                scene_to_room[str(scan_ref)] = str(room_ref)

        if not scene_to_room:
            raise RuntimeError(f"Failed to build scene->room mapping from {room_json_path}")

        cls._scene_to_room_map = scene_to_room
        return scene_to_room

    def is_same_room_and_pose(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
        *,
        trans_tol_m: float = 3.0,
        rot_tol_deg: float = 60.0,
    ) -> bool:
        """
        Return True iff:
        1) both samples belong to the same room (via `3RScan.json` grouping), and
        2) their pose is within `trans_tol_m` translation and `rot_tol_deg` rotation.

        Expected pose format is `[tx, ty, tz, qx, qy, qz, qw]`.
        """
        room_map = self._get_scene_to_room_map()

        scene_id_a = a.get("scene_id", None)
        scene_id_b = b.get("scene_id", None)
        if scene_id_a is None or scene_id_b is None:
            # We can't map hashed `scene` back to IDs reliably.
            return False

        room_a = room_map.get(str(scene_id_a))
        room_b = room_map.get(str(scene_id_b))
        if room_a is None or room_b is None or room_a != room_b:
            return False

        pose_a = a["pose"]
        pose_b = b["pose"]
        if not isinstance(pose_a, Tensor) or not isinstance(pose_b, Tensor):
            pose_a = torch.as_tensor(pose_a)
            pose_b = torch.as_tensor(pose_b)

        pose_a = pose_a.to(dtype=torch.float64)
        pose_b = pose_b.to(dtype=torch.float64)
        if pose_a.numel() != 7 or pose_b.numel() != 7:
            raise ValueError(
                f"Expected pose vectors of length 7; got {pose_a.numel()} and {pose_b.numel()}"
            )

        t_a = pose_a[:3]
        t_b = pose_b[:3]
        trans_diff = float((t_a - t_b).norm())
        if trans_diff > float(trans_tol_m):
            return False

        q_a = pose_a[3:]
        q_b = pose_b[3:]
        
        rot_diff_deg = quaternion_angle(
            q_a.detach().cpu().numpy(),
            q_b.detach().cpu().numpy(),
            degrees=True,
            normalize=True,
        )
        if rot_diff_deg > float(rot_tol_deg):
            return False

        return True

    def save_meta_parquet(
        self,
        meta_path: Optional[Union[str, Path]] = None,
        meta_file: str = "meta.parquet"
        ):

        out = (Path(meta_path) / meta_file) if meta_path is not None else (self.dataset_root / meta_file)

        out.parent.mkdir(parents=True, exist_ok=True)
        self.df.to_parquet(out, index=False)
        logger.info(f"Wrote {len(self.df):,} rows to {out}")
        return out

    @staticmethod
    def collate_fn(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Collate function that keeps paths/scenes as Python lists."""
        return {
            "idxs": torch.stack([b["idx"] for b in batch], dim=0),
            "poses": torch.stack([b["pose"] for b in batch], dim=0),
            "images_main": torch.stack([b["image_main"] for b in batch], dim=0),
            "scenes": torch.stack([torch.tensor(hash(b["scene"])) for b in batch]),
        }

