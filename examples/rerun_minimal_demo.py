from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import rerun as rr
from mmpr.data.pcd import SimplePCDLoader
from mmpr.data.camera import SimpleCameraLoader
from mmpr.modules.rerun_vis_utils import visualize_camera_pose 

# Choose the element of imported map that will be visualized
N = 100 

def parse_args() -> argparse.Namespace:
    """Function to parse arguments for rerun example

    Returns: 
        arguments: Namespace with arguments values
    
    """

    p = argparse.ArgumentParser(description="Rerun mimal demo program that visualize one pointcloud and photo")
    p.add_argument("--map-dir", type=Path, required=True, help="Path to map with pointclouds and photo files")
    p.add_argument(
        "--sensors", type=str, default="lidar,cam_fish-eye_left",
        help="Comma-separated sensors to load (subset of: lidar,cam_pinhole_left,cam_pinhole_right,cam_fish-eye_left,cam_fish-eye-right)",
    )
    p.add_argument(
        "--transform-matrix", type=Path, default="transform_to_map1.npy", help="Path to .npy file with transform matrix",
    )
    p.add_argument("--output", type=Path, required=True, help="Path to output .rrd file")
    p.add_argument("--app-id", type=str, default="mmpr_dataset_viz", help="Rerun app id")
    return p.parse_args()



def main() -> None:

    # Parsing arguments
    args = parse_args()

    # Arguments handling
    map_dir: Path = args.map_dir.resolve()
    if not map_dir.exists():
        raise FileNotFoundError(f"map_dir does not exist: {map_dir}")
    
    transform_matrix_path = args.transform_matrix.resolve()
    if (map_dir / transform_matrix_path).exists():
        transform_matrix = np.load(map_dir / transform_matrix_path)
    elif transform_matrix_path.exists():
        transform_matrix = np.load(transform_matrix_path)
    else:
        transform_matrix = None

    sensors_to_load: set[str] = set([s.strip() for s in args.sensors.split(",") if s.strip()])
    cameras = list(filter(lambda x: "cam" in x, sensors_to_load))

    # Init data loaders
    pcd_loader = SimplePCDLoader(map_dir, scans_subdir="scans", T_map_to_world=transform_matrix)
    camera_loader = SimpleCameraLoader(map_dir, cameras, T_map_to_world=transform_matrix)

    # Init Rerun and save to file
    rr.init(args.app_id)
    rr.save(str(args.output))

    # Init timestamp
    rr.set_time("time", timestamp=0)

    # Taking lidar cloud from pcd_loader ang logging it
    coords, pose, scan_path = pcd_loader[N]
    rr.log("world/lidar", rr.Transform3D(translation=pose[:3], rotation=rr.Quaternion(xyzw=pose[3:])))
    rr.log("world/lidar", rr.Points3D(coords, colors=np.array([200, 200, 0]).astype(np.uint8), radii=0.02))

    # Taking images from all mentioned cameras from camera_loader ang logging it
    images, pose, images_paths = camera_loader[N]
    for cam in cameras:
        rr.log(f"world/{cam}", rr.Image(images[cam]))
        if "fish-eye" in cam:
            visualize_camera_pose(pose=pose, fov_x_deg=140, fov_y_deg=120, depth=0.5, color=(255, 0, 200), name=cam)
        else:
            visualize_camera_pose(pose=pose, fov_x_deg=80, fov_y_deg=60, depth=1.0, color=(255, 0, 200), name=cam)

    print(f"Saved Rerun recording to {args.output}")

    # End writing session 
    rr.disconnect()



if __name__ == "__main__":
    main()