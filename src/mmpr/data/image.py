from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation
from torchvision import transforms as T
from torch import Tensor


def get_default_image_transform() -> T.Compose:
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        T.Resize([322, 322], antialias=True)
    ])

def mat44_to_pose7(Tw: np.ndarray) -> np.ndarray:
    """Convert 4x4 homogeneous transform to [tx,ty,tz,qx,qy,qz,qw]."""
    Rm = Tw[:3, :3]
    t = Tw[:3, 3]
    q = Rotation.from_matrix(Rm).as_quat()
    return np.concatenate([t, q]).astype(np.float32)


class SimpleImageLoader:
    """Load keyframe images and poses from `<map_root>/images` (default).

    Mirrors the notebook PCD loader and supports an optional transform to map1/world.
    """

    def __init__(self, map_root: Path, images_subdir: str = "zedx_front_left/rgb", T_map_to_world: Optional[np.ndarray] = None) -> None:
        self._map_root = map_root
        self._images_subdir = images_subdir
        traj_path = self._map_root / "poses.csv"
        self._poses = self._load_poses(traj_path)  # [N,7]

        if T_map_to_world is not None:
            num = self._poses.shape[0]
            out = np.zeros_like(self._poses, dtype=np.float32)
            for i in range(num):
                pose7 = self._poses[i]
                t = pose7[:3]
                q = pose7[3:]
                Rm = Rotation.from_quat(q).as_matrix().astype(np.float32)
                Tcur = np.eye(4, dtype=np.float32)
                Tcur[:3, :3] = Rm
                Tcur[:3, 3] = t
                Twi = T_map_to_world @ Tcur
                out[i] = mat44_to_pose7(Twi)
            self._poses = out

        self._image_paths = sorted(list((self._map_root / self._images_subdir).glob("*.jpg")))
        if len(self._poses) > len(self._image_paths):
            self._poses = self._poses[:len(self._image_paths)]
        if len(self._poses) != len(self._image_paths):
            raise RuntimeError(f"#images ({len(self._image_paths)}) != #poses ({len(self._poses)}) in {self._map_root}")

        self._image_transform = get_default_image_transform()

    def __len__(self) -> int:
        return len(self._image_paths)

    def __getitem__(self, idx: int) -> Tuple[Tensor, np.ndarray, Path]:
        image_path = self._image_paths[idx]
        pose7 = self._poses[idx]
        img = Image.open(image_path).convert("RGB")
        img = self._image_transform(img)
        return img, pose7, image_path

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