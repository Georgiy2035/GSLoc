from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple
from PIL import Image

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation


def mat44_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 homogeneous transform to [tx,ty,tz,qx,qy,qz,qw]."""
    Rm = T[:3, :3]
    t = T[:3, 3]
    q = Rotation.from_matrix(Rm).as_quat()
    return np.concatenate([t, q]).astype(np.float32)

def read_image_np(path: Path) -> np.ndarray:
    img = Image.open(path)
    img = img.convert("RGB")
    arr = np.asarray(img, dtype=np.uint8)  # HWC, uint8
    return arr


class SimpleCameraLoader:
    """Load keyframe camera images from `<map_dir>/keyframe_map`.
    
    Mirrors the notebook loader.
    """

    def __init__(self, map_root: Path, 
                 cameras_set: set[str] = {"cam_fish-eye_left"}, 
                 T_map_to_world: Optional[np.ndarray] = None,
                 feature_map_dir: Path = "/mnt/external_usb_hdd/6YL/Datasets/SberRobotics/mmpr_pca") -> None:
        
        self._map_root = map_root
        traj_path = self._map_root / "poses.csv"
        self._poses = self._load_poses(traj_path)
        self._cameras_set = cameras_set
        self._feature_map_dir = feature_map_dir

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

    def _camera_path_from_index(self, cam: str, index_ns: int) -> Path:
        assert cam in {"cam_pinhole_left", "cam_pinhole_right", "cam_fish-eye_left", "cam_fish-eye-right"}
        if cam == "cam_pinhole_left":
            base = self._map_root / "images"#"zedx_front_left" / "rgb"
        elif cam == "cam_pinhole_right":
            base = self._map_root / "images"#"zedx_front_right" / "rgb"
        elif cam == "cam_fish-eye_left":
            base = self._map_root / "images"#"zedxone_left" / "rgb"
        else:
            base = self._map_root / "images"#"zedxone_right" / "rgb"
        stem = "0" * (6 - len(str(index_ns))) + str(index_ns)
        for ext in ("png", "jpg", "jpeg"):
            p = base / f"{stem}.{ext}"
            if p.exists():
                return p
        # default to png
        return base / f"{stem}.png"
    
    def _feature_map_path_from_index(self, idx: int) -> Path:
        return self._feature_map_dir / f"{idx:06d}_pca_b.png"

    def __len__(self) -> int:
        return len(self._scan_paths)

    def __getitem__(self, idx: int) -> Tuple[dict[str: np.ndarray], np.ndarray, Tuple[dict[str: Path]]]:
        images = dict()
        images_paths = dict()

        for camera in self._cameras_set:
            image_path = self._camera_path_from_index(camera, idx)
            pose7 = self._poses[idx]
            images[camera] = read_image_np(image_path)
            images_paths[camera] = image_path

        feature_map_path = self._feature_map_path_from_index(idx)
        feature_map = read_image_np(feature_map_path)

        return images, pose7, images_paths, feature_map

    @staticmethod
    def _load_poses(traj_path: Path) -> np.ndarray:
        import pandas as pd

        df = pd.read_table(
            traj_path,
            header=1,
            sep=",",
            names=["timestamp", "px", "py", "pz", "qx", "qy", "qz", "qw"],
            comment="#",
        )
        vals = df[["px", "py", "pz", "qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float32)
        return vals

    


