from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from torch import Tensor

from .image import get_default_image_transform
from .pcd import load_pcd_points


def mat44_to_pose7(Tw: np.ndarray) -> np.ndarray:
    """Convert 4x4 homogeneous transform to [tx,ty,tz,qx,qy,qz,qw]."""
    Rm = Tw[:3, :3]
    t = Tw[:3, 3]
    q = Rotation.from_matrix(Rm).as_quat()
    return np.concatenate([t, q]).astype(np.float32)


class SimpleMultimodalLoader:
    """Load paired keyframe images and scans, with poses from `<map_root>/poses.csv`.

    Assumes both images and scans correspond to the same sequence and ordering.
    If counts differ across modalities or poses, truncates all to the minimum length.
    Supports an optional transform to re-express poses in a different world/map frame.
    """

    def __init__(
        self,
        map_root: Path,
        images_subdir: str = "zedx_front_left/rgb",
        scans_subdir: str = "scans",
        T_map_to_world: Optional[np.ndarray] = None,
        pcd_voxel_size: float | None = None,
    ) -> None:
        self._map_root = map_root
        self._images_subdir = images_subdir
        self._scans_subdir = scans_subdir
        self._pcd_voxel_size = pcd_voxel_size

        traj_path = self._map_root / "poses.csv"
        poses = self._load_poses(traj_path)  # [N,7]

        if T_map_to_world is not None:
            num = poses.shape[0]
            out = np.zeros_like(poses, dtype=np.float32)
            for i in range(num):
                pose7 = poses[i]
                t = pose7[:3]
                q = pose7[3:]
                Rm = Rotation.from_quat(q).as_matrix().astype(np.float32)
                Tcur = np.eye(4, dtype=np.float32)
                Tcur[:3, :3] = Rm
                Tcur[:3, 3] = t
                Twi = T_map_to_world @ Tcur
                out[i] = mat44_to_pose7(Twi)
            poses = out

        image_paths = sorted(list((self._map_root / self._images_subdir).glob("*.jpg")))
        scan_paths = sorted(list((self._map_root / self._scans_subdir).glob("*.pcd")))

        # Align counts by truncating to the minimum length across the three sources.
        n = min(len(image_paths), len(scan_paths), len(poses))
        if n == 0:
            raise RuntimeError(
                f"No paired data found under {self._map_root} "
                f"(images={len(image_paths)}, scans={len(scan_paths)}, poses={len(poses)})"
            )
        if len(image_paths) != n:
            image_paths = image_paths[:n]
        if len(scan_paths) != n:
            scan_paths = scan_paths[:n]
        if len(poses) != n:
            poses = poses[:n]

        self._poses = poses
        self._image_paths = image_paths
        self._scan_paths = scan_paths
        self._image_transform = get_default_image_transform()

    def __len__(self) -> int:
        return len(self._poses)

    def __getitem__(self, idx: int) -> Tuple[Tensor, np.ndarray, np.ndarray, Path, Path]:
        """Return (image_tensor, pointcloud_xyz, pose7, image_path, scan_path)."""
        image_path = self._image_paths[idx]
        scan_path = self._scan_paths[idx]
        pose7 = self._poses[idx]

        img = Image.open(image_path).convert("RGB")
        img_tensor = self._image_transform(img)

        points = load_pcd_points(scan_path, voxel_size=self._pcd_voxel_size)
        return img_tensor, points, pose7, image_path, scan_path

    @staticmethod
    def _load_poses(traj_path: Path) -> np.ndarray:
        import pandas as pd

        df = pd.read_table(
            traj_path,
            header=None,
            sep=",",
            names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
            comment="#",
        )
        vals = df[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float32)
        return vals
