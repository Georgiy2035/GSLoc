#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import numpy as np
import rerun as rr  # pip install rerun-sdk
import torch

from mmpr.data.data_reader import FramesDataReader


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize multimodal dataset with Rerun and save recording")
    p.add_argument("--mav0-dir", type=Path, required=True, help="Path to mav0 directory")
    p.add_argument("--frames-csv", type=Path, default=None, help="Optional custom path to frames.csv")
    p.add_argument(
        "--sensors", type=str, default="lidar_joined,cam0,cam1,cam2,cam3,depth",
        help="Comma-separated sensors to load (subset of: lidar_joined,lidar0,lidar1,cam0,cam1,cam2,cam3,depth)",
    )
    p.add_argument("--voxel", type=float, default=0.1, help="Quantization size for pointclouds")
    p.add_argument("--stride", type=int, default=1, help="Use every N-th frame for logging")
    p.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames (0 = all)")
    p.add_argument("--output", type=Path, required=True, help="Path to output .rrd file")
    p.add_argument("--app-id", type=str, default="mmpr_dataset_viz", help="Rerun app id")
    return p.parse_args()


def lidar_entity_root(name: str) -> str:
    return f"world/{name}"


def camera_entity_root(name: str) -> str:
    return f"world/{name}"


def main() -> None:
    args = parse_args()

    mav0_dir: Path = args.mav0_dir.resolve()
    frames_csv = args.frames_csv if args.frames_csv is not None else (mav0_dir / "frames.csv")
    sensors_to_load: set[str] = set([s.strip() for s in args.sensors.split(",") if s.strip()])

    # Initialize reader
    reader = FramesDataReader(
        mav0_dir=mav0_dir,
        frames_csv=frames_csv,
        pointcloud_quantization_size=args.voxel,
        sensors_to_load=sensors_to_load,
        image_transform=None,  # default ToTensorV2 if available
    )

    # Init Rerun and save to file
    rr.init(args.app_id)
    rr.save(str(args.output))

    # Log static world/trajectory as a line for context
    poses_np = reader._poses  # [N,7]
    traj_xyz = poses_np[:, :3].astype(np.float32)
    rr.log("world/trajectory", rr.LineStrips3D(traj_xyz.reshape(1, -1, 3), colors=[255, 255, 0], radii=0.02), static=True)

    # For each frame
    num_frames = len(reader)
    limit = args.max_frames if args.max_frames and args.max_frames > 0 else num_frames
    count = 0
    for idx in range(0, num_frames, args.stride):
        if count >= limit:
            break
        item = reader[idx]

        # Use pose timestamp as the primary timeline
        ts = int(item["pose_timestamp"].item()) * 1e-9  # ns -> seconds
        rr.set_time("time", timestamp=ts)

        # Pose as a Transform3D at world/query
        x, y, z, qx, qy, qz, qw = item["pose"].tolist()
        rr.log("world/pose", rr.Transform3D(translation=[x, y, z], rotation=rr.Quaternion(xyzw=[qx, qy, qz, qw])))

        # LiDARs
        for lidar_name in ("lidar_joined", "lidar0", "lidar1"):
            ts_key = f"timestamp_{lidar_name}"
            coords_key = f"pointcloud_{lidar_name}_coords"
            feats_key = f"pointcloud_{lidar_name}_feats"
            if ts_key in item and coords_key in item and feats_key in item:
                coords = item[coords_key].numpy()
                feats = item[feats_key].numpy()
                # Project intensity (if any) to grayscale color
                colors = None
                if feats.size > 0:
                    f = feats.squeeze(-1)
                    f = (255.0 * (f - f.min()) / (f.max() - f.min() + 1e-6)).clip(0, 255).astype(np.uint8)
                    colors = np.stack([f, f, f], axis=-1)
                rr.log(lidar_entity_root(lidar_name), rr.Points3D(coords, colors=colors, radii=0.02))

        # Cameras
        for cam in ("cam0", "cam1", "cam2", "cam3"):
            t_key = f"timestamp_{cam}"
            img_key = f"image_{cam}"
            if t_key in item and img_key in item:
                img_chw = item[img_key]
                # Convert CHW float [0,1] to HWC u8 for logging
                img_hwc = (img_chw.clamp(0, 1) * 255.0).byte().permute(1, 2, 0).cpu().numpy()
                rr.log(camera_entity_root(cam), rr.Image(img_hwc))

        # Depth
        if "depth_timestamp" in item and "depth" in item:
            depth = item["depth"].cpu().numpy()
            # Map depth to colormap-like grayscale
            d = depth
            d_min, d_max = float(np.nanmin(d)), float(np.nanmax(d))
            d_vis = (255.0 * (d - d_min) / (d_max - d_min + 1e-6)).clip(0, 255).astype(np.uint8)
            rr.log("world/depth", rr.Image(d_vis))

        count += 1

    print(f"Saved Rerun recording to {args.output}")


if __name__ == "__main__":
    main()


