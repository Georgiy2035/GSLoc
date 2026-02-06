from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def mat44_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 homogeneous transform to [tx,ty,tz,qx,qy,qz,qw]."""
    Rm = T[:3, :3]
    t = T[:3, 3]
    q = Rotation.from_matrix(Rm).as_quat()
    return np.concatenate([t, q]).astype(np.float32)


def load_pcd_points(path: Path, voxel_size: float | None = None) -> np.ndarray:
    pc = o3d.io.read_point_cloud(str(path))
    if voxel_size is not None and voxel_size > 0.0:
        pc = pc.voxel_down_sample(float(voxel_size))
    pts = np.asarray(pc.points, dtype=np.float32)
    if pts.size == 0:
        return pts.reshape(0, 3).astype(np.float32)
    return pts.astype(np.float32)


class SimplePCDLoader:
    """Load keyframe scans and poses from `<map_dir>/keyframe_map`.
    
    Mirrors the notebook loader and supports an optional transform to map1/world.
    """

    def __init__(self, map_root: Path, scans_subdir: str = "scans", T_map_to_world: Optional[np.ndarray] = None, header: int = 0) -> None:
        self._map_root = map_root
        self._scans_subdir = scans_subdir
        traj_path = self._map_root / "poses.csv"
        
        self._poses = self._load_poses(traj_path, header=header)  # [N,7]

        if T_map_to_world is not None:
            num = self._poses.shape[0]
            out = np.zeros_like(self._poses, dtype=np.float32)
            for i in range(num):
                pose7 = self._poses[i]
                t = pose7[:3]
                q = pose7[3:]
                Rm = Rotation.from_quat(q).as_matrix().astype(np.float32)
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = Rm
                T[:3, 3] = t
                Twi = T_map_to_world @ T
                out[i] = mat44_to_pose7(Twi)
            self._poses = out

        self._scan_paths = sorted(list((self._map_root / self._scans_subdir).glob("*.pcd")))
        
        # if len(self._scan_paths) != len(self._poses):
        #     raise RuntimeError(f"#scans ({len(self._scan_paths)}) != #poses ({len(self._poses)}) in {self._map_root}")

    def __len__(self) -> int:
        return len(self._scan_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, Path]:
        scan_path = self._scan_paths[idx]
        pose7 = self._poses[idx]
        points = load_pcd_points(scan_path)
        return points, pose7, scan_path

    @staticmethod
    def _load_poses(traj_path: Path, header: int = 0) -> np.ndarray:
        import pandas as pd
        df = pd.read_table(
            traj_path,
            header=header,
            sep=",",
            names=["timestamp", "px", "py", "pz", "qx", "qy", "qz", "qw"],
            comment="#",
        )
        vals = df[["px", "py", "pz", "qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float32)
        return vals


