from __future__ import annotations

from pathlib import Path
from collections.abc import Iterable

import numpy as np
import pandas as pd
import open3d as o3d
from PIL import Image
import torch
from torch import Tensor
import MinkowskiEngine as ME

# Optional Albumentations at module import
try:
    import albumentations as A  # type: ignore
    from albumentations.pytorch import ToTensorV2  # type: ignore
except Exception:  # optional dependency
    A = None  # type: ignore
    ToTensorV2 = None  # type: ignore
ToTensorV2 = None

class MmprFramesDataReader:
    """
    Dataset/DataLoader friendly reader for frames.csv-based multimodal dataset.

    Expected CSV columns (minimal):
      - ts, tx, ty, tz, qx, qy, qz, qw

    LiDAR file locations:
      - lidar:       <map_dir>/scans/<timestamp>.pcd

    Returned item structure:
      {
        "pose": Tensor[7],  # [px, py, pz, qx, qy, qz, qw]
        "pose_timestamp": Tensor[1],
        # Optional LiDAR(s):
        "timestamp_lidar": Tensor[1], "pointcloud_lidar_coords": Tensor[N, 3], "pointcloud_lidar_feats": Tensor[N, 1],
      }

    Collate function applies MinkowskiEngine quantization and batching.
    """

    def __init__(
        self,
        map_dir: str | Path,
        poses_csv: str | Path | None = None,
        pointcloud_quantization_size: float = 0.05,
        image_transform: object | None = None,
        sensors_to_load: Iterable[str] | None = None,
    ) -> None:
        map_dir = Path(map_dir)
        if not map_dir.exists():
            raise FileNotFoundError(f"map_dir does not exist: {map_dir}")

        self._map_dir = map_dir
        self._poses_csv = Path(poses_csv) if poses_csv is not None else (map_dir / "poses.csv")
        if not self._poses_csv.exists():
            raise FileNotFoundError(f"frames.csv not found at {self._poses_csv}")

        self._df = self._read_frames_csv(self._poses_csv)
        self._pointcloud_quantization_size = pointcloud_quantization_size

        # Sensors selection (include lidar sensors)
        default_sensors = {"lidar", "cam_pinhole_left", "cam_pinhole_right", "cam_fish-eye_left", "cam_fish-eye-right"}
        self._sensors_to_load = set(sensors_to_load) if sensors_to_load is not None else default_sensors

        # Precompute pose columns
        self._poses: np.ndarray = self._df[["px", "py", "pz", "qx", "qy", "qz", "qw"]].to_numpy()
        self._pose_ts: np.ndarray = self._df["ts"].astype("float32").to_numpy()

        # LiDAR setup (may include multiple sensors)
        self._lidar_names: list[str] = [
            name for name in ("lidar",)
            if name in self._sensors_to_load
        ]
        self._lidar_ids_map: dict[str, np.ndarray] = {}
        self._scan_paths_map: dict[str, list[Path]] = {}
        for name in self._lidar_names:
            ids_arr = np.array(range(len(self._df)))
            self._lidar_ids_map[name] = ids_arr
            paths = [self._lidar_path_from_index(name, int(id)) for id in ids_arr]
            for p in paths:
                if not p.exists():
                    raise FileNotFoundError(f"Missing LiDAR scan file: {p}")
            self._scan_paths_map[name] = paths

        # Optional modalities
        self._camera_names: list[str] = [
            name for name in ("cam_pinhole_left", "cam_pinhole_right", "cam_fish-eye_left", "cam_fish-eye-right") if name in self._sensors_to_load
        ]

        # Albumentations transform pipeline for images
        self._image_transform = image_transform
        if self._camera_names and self._image_transform is None and A is not None and ToTensorV2 is not None:
            self._image_transform = A.Compose([ToTensorV2()])

    def __len__(self) -> int:
        return len(self._df)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")

        pose = torch.tensor(self._poses[idx], dtype=torch.float32)
        pose_ts = torch.tensor([self._pose_ts[idx]], dtype=torch.float32)

        item: dict[str, Tensor] = {
            "pose": pose,
            "pose_timestamp": pose_ts,
            "idx": idx
        }

        # LiDARs (optional, multiple)
        for name in self._lidar_names:
            coords_np, feats_np = self._read_pointcloud(self._scan_paths_map[name][idx])
            coords = torch.from_numpy(coords_np.astype(np.float32))
            feats = torch.from_numpy(feats_np.astype(np.float32))
            item[f"pointcloud_{name}_coords"] = coords
            item[f"pointcloud_{name}_feats"] = feats

        # Cameras
        if self._camera_names:
            for cam in self._camera_names:
                img_path = self._camera_path_from_index(cam, idx)
                if not img_path.exists():
                    raise FileNotFoundError(f"Missing image for {cam}: {img_path}")
                img_np = self._read_image_np(img_path)
                # Albumentations pipeline expects dict with 'image'
                if self._image_transform is not None:
                    transformed = self._image_transform(image=img_np)
                    img_t = transformed["image"]
                    # Ensure dtype float32 CHW
                    if not isinstance(img_t, torch.Tensor):
                        img_t = torch.from_numpy(np.asarray(img_t))
                    if img_t.ndim == 3 and img_t.shape[-1] in (1, 3):
                        img_t = img_t.permute(2, 0, 1).contiguous()
                    if img_t.dtype != torch.float32:
                        img_t = img_t.float()
                    # If values are 0..255, normalize to 0..1
                    if img_t.max() > 1.5:
                        img_t = img_t / 255.0
                else:
                    img_t = self._to_tensor_chw(img_np)

                item[f"image_{cam}"] = img_t

        return item

    def collate_fn(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor | list[Tensor]]:
        poses = torch.stack([item["pose"] for item in batch])
        pose_timestamps = torch.stack([item["pose_timestamp"] for item in batch]).squeeze(-1)
        pose_ids = torch.stack([item["idx"] for item in batch]).squeeze(-1)

        out: dict[str, Tensor | list[Tensor]] = {
            "poses": poses,
            "pose_timestamps": pose_timestamps,
            "idx": pose_ids
        }

        # LiDARs in collate (optional, per-sensor)
        for name in getattr(self, "_lidar_names", []):
            coords_key = f"pointcloud_{name}_coords"
            feats_key = f"pointcloud_{name}_feats"
            if all((coords_key in e and feats_key in e) for e in batch):
                coords_list = [e[coords_key] for e in batch]
                feats_list = [e[feats_key] for e in batch]

                q_coords_list: list[Tensor] = []
                q_feats_list: list[Tensor] = []
                for coordinates, features in zip(coords_list, feats_list):
                    q_coords, q_feats = ME.utils.sparse_quantize(
                        coordinates=coordinates,
                        features=features,
                        quantization_size=self._pointcloud_quantization_size,
                    )
                    q_coords_list.append(q_coords)
                    q_feats_list.append(q_feats)

                out[f"pointclouds_{name}_coords"] = ME.utils.batched_coordinates(q_coords_list)
                out[f"pointclouds_{name}_feats"] = torch.cat(q_feats_list, dim=0)

        # Pass-through camera images as lists (variable sizes), with stacked timestamps
        if self._camera_names:
            for cam in self._camera_names:
                out[f"images_{cam}"] = [e[f"image_{cam}"] for e in batch]

        return out

    def _read_frames_csv(self, path: Path) -> pd.DataFrame:
        dtype_map = {
            "ts": np.float32,
            "px": np.float64,
            "py": np.float64,
            "pz": np.float64,
            "qx": np.float64,
            "qy": np.float64,
            "qz": np.float64,
            "qw": np.float64,
        }

        df = pd.read_csv(path, dtype=dtype_map)
        return df

    def _lidar_path_from_index(self, sensor_name: str, index_ns: int) -> Path:
        filename = "0" * (6 - len(str(index_ns))) + f"{index_ns}.pcd"
        if sensor_name == "lidar":
            return self._map_dir / "scans" / filename
        raise ValueError(f"Unknown lidar sensor: {sensor_name}")

    def _read_pointcloud(self, filepath: Path) -> tuple[np.ndarray, np.ndarray]:
        scan = o3d.io.read_point_cloud(str(filepath))
        if not scan.has_points():
            raise ValueError(f"Scan file {filepath} is empty or invalid.")

        points = np.asarray(scan.points)
        if points.size == 0:
            raise ValueError(f"Scan file {filepath} contains zero points.")

        coordinates = points[:, :3].astype(np.float32)
        if points.shape[1] >= 4:
            features = points[:, 3:4].astype(np.float32)
        else:
            features = np.ones((coordinates.shape[0], 1), dtype=np.float32)
        return coordinates, features

    def _camera_path_from_index(self, cam: str, index_ns: int) -> Path:
        assert cam in {"cam_pinhole_left", "cam_pinhole_right", "cam_fish-eye_left", "cam_fish-eye-right"}
        if cam == "cam_pinhole_left":
            base = self._map_dir / "zedx_front_left" / "rgb"
        elif cam == "cam_pinhole_right":
            base = self._map_dir / "zedx_front_right" / "rgb"
        elif cam == "cam_fish-eye_left":
            base = self._map_dir / "zedxone_left" / "rgb"
        else:
            base = self._map_dir / "zedxone_right" / "rgb"
        stem = "0" * (6 - len(str(index_ns))) + str(index_ns)
        for ext in ("png", "jpg", "jpeg"):
            p = base / f"{stem}.{ext}"
            if p.exists():
                return p
        # default to png
        return base / f"{stem}.png"

    def _read_image_np(self, path: Path) -> np.ndarray:
        img = Image.open(path)
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.uint8)  # HWC, uint8
        return arr

    def _to_tensor_chw(self, img_hwc: np.ndarray) -> Tensor:
        # Convert HWC uint8 -> CHW float32 in [0,1]
        t = torch.from_numpy(img_hwc).permute(2, 0, 1).contiguous().float()
        if t.max() > 1.5:
            t = t / 255.0
        return t