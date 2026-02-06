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


class FramesDataReader:
    """
    Dataset/DataLoader friendly reader for frames.csv-based multimodal dataset.

    Expected CSV columns (minimal):
      - pose_ts, x, y, z, qx, qy, qz, qw
      - lidar_joined_ts and/or lidar0_ts and/or lidar1_ts (any subset)

    LiDAR file locations:
      - lidar_joined: <mav0_dir>/lidar_joined/data/<timestamp>.pcd
      - lidar0:       <mav0_dir>/lidar0/<timestamp>.pcd
      - lidar1:       <mav0_dir>/lidar1/<timestamp>.pcd

    Returned item structure:
      {
        "pose": Tensor[7],  # [x, y, z, qx, qy, qz, qw]
        "pose_timestamp": Tensor[1],
        # Optional LiDAR(s):
        "timestamp_lidar_joined": Tensor[1], "pointcloud_lidar_joined_coords": Tensor[N, 3], "pointcloud_lidar_joined_feats": Tensor[N, 1],
        "timestamp_lidar0": Tensor[1],       "pointcloud_lidar0_coords": Tensor[N, 3],       "pointcloud_lidar0_feats": Tensor[N, 1],
        "timestamp_lidar1": Tensor[1],       "pointcloud_lidar1_coords": Tensor[N, 3],       "pointcloud_lidar1_feats": Tensor[N, 1],
        # Optional modalities if available in frames.csv and enabled:
        "timestamp_cam0": Tensor[1], "image_cam0": Tensor[C, H, W] (float32 in [0,1] by default),
        "timestamp_cam1": Tensor[1], "image_cam1": Tensor[C, H, W],
        "timestamp_cam2": Tensor[1], "image_cam2": Tensor[C, H, W],
        "timestamp_cam3": Tensor[1], "image_cam3": Tensor[C, H, W],
        "depth_timestamp": Tensor[1], "depth": Tensor[H, W] (float32),
      }

    Collate function applies MinkowskiEngine quantization and batching.
    """

    def __init__(
        self,
        mav0_dir: str | Path,
        frames_csv: str | Path | None = None,
        pointcloud_quantization_size: float = 0.05,
        image_transform: object | None = None,
        sensors_to_load: Iterable[str] | None = None,
    ) -> None:
        mav0_dir = Path(mav0_dir)
        if not mav0_dir.exists():
            raise FileNotFoundError(f"mav0_dir does not exist: {mav0_dir}")

        self._mav0_dir = mav0_dir
        self._frames_csv = Path(frames_csv) if frames_csv is not None else (mav0_dir / "frames.csv")
        if not self._frames_csv.exists():
            raise FileNotFoundError(f"frames.csv not found at {self._frames_csv}")

        self._df = self._read_frames_csv(self._frames_csv)
        self._pointcloud_quantization_size = pointcloud_quantization_size

        # Sensors selection (include lidar sensors)
        default_sensors = {"lidar_joined", "lidar0", "lidar1", "cam0", "cam1", "cam2", "cam3", "depth"}
        self._sensors_to_load = set(sensors_to_load) if sensors_to_load is not None else default_sensors

        # Precompute pose columns
        self._poses: np.ndarray = self._df[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy()
        self._pose_ts: np.ndarray = self._df["pose_ts"].astype("int64").to_numpy()

        # LiDAR setup (may include multiple sensors)
        self._lidar_names: list[str] = [
            name for name in ("lidar_joined", "lidar0", "lidar1")
            if f"{name}_ts" in self._df.columns and name in self._sensors_to_load
        ]
        self._lidar_ts_map: dict[str, np.ndarray] = {}
        self._scan_paths_map: dict[str, list[Path]] = {}
        for name in self._lidar_names:
            ts_arr = self._df[f"{name}_ts"].astype("int64").to_numpy()
            self._lidar_ts_map[name] = ts_arr
            paths = [self._lidar_path_from_timestamp(name, int(ts)) for ts in ts_arr]
            for p in paths:
                if not p.exists():
                    raise FileNotFoundError(f"Missing LiDAR scan file for {name}: {p}")
            self._scan_paths_map[name] = paths

        # Optional modalities
        self._camera_names: list[str] = [
            name for name in ("cam0", "cam1", "cam2", "cam3") if f"{name}_ts" in self._df.columns and name in self._sensors_to_load
        ]
        self._has_depth: bool = ("depth_ts" in self._df.columns) and ("depth" in self._sensors_to_load)

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
        pose_ts = torch.tensor([self._pose_ts[idx]], dtype=torch.int64)

        item: dict[str, Tensor] = {
            "pose": pose,
            "pose_timestamp": pose_ts,
        }

        # LiDARs (optional, multiple)
        for name in self._lidar_names:
            lidar_ts_val = int(self._lidar_ts_map[name][idx])
            lidar_ts = torch.tensor([lidar_ts_val], dtype=torch.int64)
            coords_np, feats_np = self._read_pointcloud(self._scan_paths_map[name][idx])
            coords = torch.from_numpy(coords_np.astype(np.float32))
            feats = torch.from_numpy(feats_np.astype(np.float32))
            item[f"timestamp_{name}"] = lidar_ts
            item[f"pointcloud_{name}_coords"] = coords
            item[f"pointcloud_{name}_feats"] = feats

        # Cameras
        if self._camera_names:
            for cam in self._camera_names:
                ts_val = int(self._df.iloc[idx][f"{cam}_ts"])  # frames.csv guaranteed complete
                cam_ts = torch.tensor([ts_val], dtype=torch.int64)
                img_path = self._camera_path_from_timestamp(cam, ts_val)
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

                item[f"timestamp_{cam}"] = cam_ts
                item[f"image_{cam}"] = img_t

        # Depth
        if self._has_depth:
            ts_val = int(self._df.iloc[idx]["depth_ts"])  # frames.csv guaranteed complete
            depth_ts = torch.tensor([ts_val], dtype=torch.int64)
            depth_path = self._depth_path_from_timestamp(ts_val)
            if not depth_path.exists():
                raise FileNotFoundError(f"Missing depth map: {depth_path}")
            depth = self._read_depth(depth_path)
            item["depth_timestamp"] = depth_ts
            item["depth"] = depth

        return item

    def collate_fn(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor | list[Tensor]]:
        poses = torch.stack([item["pose"] for item in batch])
        pose_timestamps = torch.stack([item["pose_timestamp"] for item in batch]).squeeze(-1)

        out: dict[str, Tensor | list[Tensor]] = {
            "poses": poses,
            "pose_timestamps": pose_timestamps,
        }

        # LiDARs in collate (optional, per-sensor)
        for name in getattr(self, "_lidar_names", []):
            ts_key = f"timestamp_{name}"
            coords_key = f"pointcloud_{name}_coords"
            feats_key = f"pointcloud_{name}_feats"
            if all((ts_key in e and coords_key in e and feats_key in e) for e in batch):
                lidar_timestamps = torch.stack([e[ts_key] for e in batch]).squeeze(-1)
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

                out[f"timestamps_{name}"] = lidar_timestamps
                out[f"pointclouds_{name}_coords"] = ME.utils.batched_coordinates(q_coords_list)
                out[f"pointclouds_{name}_feats"] = torch.cat(q_feats_list, dim=0)

        # Pass-through camera images as lists (variable sizes), with stacked timestamps
        if self._camera_names:
            for cam in self._camera_names:
                out[f"images_{cam}"] = [e[f"image_{cam}"] for e in batch]
                out[f"timestamps_{cam}"] = torch.stack([e[f"timestamp_{cam}"] for e in batch]).squeeze(-1)

        if self._has_depth:
            out["depths"] = [e["depth"] for e in batch]
            out["depth_timestamps"] = torch.stack([e["depth_timestamp"] for e in batch]).squeeze(-1)

        return out

    def _read_frames_csv(self, path: Path) -> pd.DataFrame:
        dtype_map = {
            "pose_ts": np.int64,
            "x": np.float64,
            "y": np.float64,
            "z": np.float64,
            "qx": np.float64,
            "qy": np.float64,
            "qz": np.float64,
            "qw": np.float64,
        }
        # Timestamps may have missing values in general, but our frames.csv is cleaned; still read as Int64
        for col in [
            "lidar0_ts",
            "lidar1_ts",
            "lidar_joined_ts",
            "cam0_ts",
            "cam1_ts",
            "cam2_ts",
            "cam3_ts",
            "depth_ts",
        ]:
            dtype_map[col] = "Int64"

        df = pd.read_csv(path, dtype=dtype_map)
        return df

    def _lidar_path_from_timestamp(self, sensor_name: str, timestamp_ns: int) -> Path:
        filename = f"{timestamp_ns:018d}.pcd"
        if sensor_name == "lidar_joined":
            return self._mav0_dir / "lidar_joined" / "data" / filename
        elif sensor_name == "lidar0":
            return self._mav0_dir / "lidar0" / filename
        elif sensor_name == "lidar1":
            return self._mav0_dir / "lidar1" / filename
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

    def _camera_path_from_timestamp(self, cam: str, timestamp_ns: int) -> Path:
        assert cam in {"cam0", "cam1", "cam2", "cam3"}
        base = self._mav0_dir / cam / "data"
        stem = f"{timestamp_ns:018d}"
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

    def _depth_path_from_timestamp(self, timestamp_ns: int) -> Path:
        base = self._mav0_dir / "depth" / "data"
        stem = f"{timestamp_ns:018d}"
        for ext in ("npy", "npz"):
            p = base / f"{stem}.{ext}"
            if p.exists():
                return p
        return base / f"{stem}.npy"

    def _read_depth(self, path: Path) -> Tensor:
        if path.suffix == ".npy":
            depth = np.load(path)
        elif path.suffix == ".npz":
            data = np.load(path)
            # take the first array in the archive
            key = list(data.keys())[0]
            depth = data[key]
        else:
            raise ValueError(f"Unsupported depth format: {path.suffix}")
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        return torch.from_numpy(depth)



class LidarScansReader:
    """Simplified reader for keyframe_map datasets with only LiDAR scans and poses.

    This reader expects a directory structure like:
      - <keyframe_map_dir>/scans/*.pcd
      - <poses_csv> file with columns: ts, px, py, pz, qx, qy, qz, qw
        Optionally, the CSV may include a leading filename/index column (e.g.,
        scan basename with or without the .pcd extension). When present, it is
        used to map rows directly to files inside the scans directory.

    Returned item structure (per index):
      {
        "pose": Tensor[7],  # [x, y, z, qx, qy, qz, qw]
        "pose_timestamp": Tensor[1],
        "pointcloud_lidar_coords": Tensor[N, 3],
        "pointcloud_lidar_feats": Tensor[N, 1],
      }

    The collate_fn applies MinkowskiEngine quantization and batching with the
    same logic as the full reader.
    """

    def __init__(self, keyframe_map_dir: str | Path, poses_csv: str | Path, max_point_distance: float = 30.0) -> None:
        """Initialize the reader.

        Args:
            keyframe_map_dir: Path to the keyframe_map directory that contains `scans/`.
            poses_csv: Path to the poses.csv file.
        """
        base_dir = Path(keyframe_map_dir)
        if not base_dir.exists():
            raise FileNotFoundError(f"keyframe_map_dir does not exist: {base_dir}")

        scans_dir = base_dir / "scans"
        if not scans_dir.exists():
            raise FileNotFoundError(f"Missing scans directory: {scans_dir}")

        poses_csv_path = Path(poses_csv)
        if not poses_csv_path.exists():
            raise FileNotFoundError(f"poses.csv not found at {poses_csv_path}")

        self._base_dir = base_dir
        self._scans_dir = scans_dir
        self._poses_csv = poses_csv_path
        self._max_point_distance = max_point_distance

        df = self._read_poses_csv(poses_csv_path)
        # Store poses in the same [x, y, z, qx, qy, qz, qw] order as FramesDataReader
        self._poses: np.ndarray = df[["px", "py", "pz", "qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float64)
        # Pose timestamps as float64 (dataset typically stores seconds); keep as int64 tensor later if needed
        self._pose_ts: np.ndarray = df["ts"].to_numpy(dtype=np.float64)

        # Determine scan file paths either from an explicit first column (scan name)
        # or by sorted listing with strict length match.
        self._scan_paths: list[Path] = []
        for name in df["scan_name"].tolist():
            filename = f"{name:06d}.pcd"
            p = self._scans_dir / filename
            if not p.exists():
                raise FileNotFoundError(f"Missing LiDAR scan file referenced by CSV: {p}")
            self._scan_paths.append(p)
        self._pointcloud_quantization_size: float = 0.05

    def __len__(self) -> int:
        """Return the number of frames."""
        return len(self._scan_paths)

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        """Return a sample consisting of pose and LiDAR point cloud.

        Args:
            idx: Sample index.

        Returns:
            A dictionary with keys: "pose", "pose_timestamp",
            "pointcloud_lidar_coords", "pointcloud_lidar_feats".
        """
        if idx < 0 or idx >= len(self):
            raise IndexError("Index out of range.")

        pose = torch.tensor(self._poses[idx], dtype=torch.float32)
        pose_ts = torch.tensor([self._pose_ts[idx]], dtype=torch.float32)

        coords_np, feats_np = self._read_pointcloud(self._scan_paths[idx])
        coords = torch.from_numpy(coords_np.astype(np.float32))
        feats = torch.from_numpy(feats_np.astype(np.float32))

        return {
            "pose": pose,
            "pose_timestamp": pose_ts,
            "pointcloud_lidar_coords": coords,
            "pointcloud_lidar_feats": feats,
        }

    def collate_fn(self, batch: list[dict[str, Tensor]]) -> dict[str, Tensor]:
        """Collate function with MinkowskiEngine quantization/batching for LiDAR.

        Args:
            batch: List of per-item dicts as returned by __getitem__.

        Returns:
            A dictionary with batched poses/timestamps and quantized, batched point clouds.
        """
        poses = torch.stack([item["pose"] for item in batch])
        pose_timestamps = torch.stack([item["pose_timestamp"] for item in batch]).squeeze(-1)

        coords_list = [e["pointcloud_lidar_coords"] for e in batch]
        feats_list = [e["pointcloud_lidar_feats"] for e in batch]

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

        return {
            "poses": poses,
            "pose_timestamps": pose_timestamps,
            "pointclouds_lidar_coords": ME.utils.batched_coordinates(q_coords_list),
            "pointclouds_lidar_feats": torch.cat(q_feats_list, dim=0),
        }

    def _read_poses_csv(self, path: Path) -> pd.DataFrame:
        """Read poses.csv supporting optional leading scan-name column and comments.

        Supported layouts:
          - 9 columns: scan_name, ts, px, py, pz, qx, qy, qz, qw

        Lines starting with '#' are ignored.
        """
        df = pd.read_csv(path)
        return df

    def _read_pointcloud(self, filepath: Path) -> tuple[np.ndarray, np.ndarray]:
        """Read a PCD file and return coordinates and features arrays.

        If intensity or additional channels are present, use the first as a single-channel
        feature. Otherwise, features default to ones.
        """
        scan = o3d.io.read_point_cloud(str(filepath))
        if not scan.has_points():
            raise ValueError(f"Scan file {filepath} is empty or invalid.")

        points = np.asarray(scan.points)
        if points.size == 0:
            raise ValueError(f"Scan file {filepath} contains zero points.")

        if self._max_point_distance is not None:
            distances = np.linalg.norm(points[:, :3], axis=1)
            points = points[distances <= self._max_point_distance]

        coordinates = points[:, :3].astype(np.float32)

        # Try to use intensity if present; otherwise default to ones.
        if scan.has_colors():
            # Convert RGB intensity to a single channel by average
            colors = np.asarray(scan.colors, dtype=np.float32)
            intensity = colors.mean(axis=1, keepdims=True)
            features = intensity.astype(np.float32)
        else:
            features = np.ones((coordinates.shape[0], 1), dtype=np.float32)

        return coordinates, features
