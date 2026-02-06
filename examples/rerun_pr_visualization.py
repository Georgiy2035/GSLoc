import argparse
from pathlib import Path
import rerun as rr
import os
from mmpr.modules.rerun_vis_utils import write_rerun_visualization, PR_results_visual
from mmpr.inference.data import LocalizationResult, LocalizedCandidate

def parse_args() -> argparse.Namespace:
    """Function to parse arguments for rerun example

    Returns: 
        arguments: Namespace with arguments values
    
    """

    p = argparse.ArgumentParser(description="Rerun mimal demo program that visualize one pointcloud and photo")
    p.add_argument(
        "--db-map-dir", type=Path, required=True, 
        help="Path to databse map with pointclouds and photo files")
    p.add_argument(
        "--q-map-dir", type=Path, required=True,
        help="Path to query map with pointclouds and photo files")
    p.add_argument(
        "--base-map-dir", type=Path, default="",
        help="Path to pointcloud base map")
    p.add_argument(
        "--results-dir", type=Path, required=True, 
        help="Path to directory with JSON files that collect information about query-database matching")
    p.add_argument(
        "--db-feature-map-dir", type=Path, default="/mnt/external_usb_hdd/6YL/Datasets/SberRobotics/mmpr_pca/map1_pca/pca", 
        help="Path to directory with database feature maps from MegaLoc last layers")
    p.add_argument(
        "--q-feature-map-dir", type=Path, default="/mnt/external_usb_hdd/6YL/Datasets/SberRobotics/mmpr_pca/map3_pca/pca", 
        help="Path to directory with query feature maps from MegaLoc last layers")
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
    """Data visualization with rerun"""

    #arguments parsing
    args = parse_args()
    db_map_dir: Path = args.db_map_dir.resolve()
    q_map_dir: Path = args.q_map_dir.resolve()
    db_feature_map_dir: Path = args.db_feature_map_dir.resolve()
    q_feature_map_dir: Path = args.q_feature_map_dir.resolve()
    if args.base_map_dir == "":
        base_map_dir = None
    else:
        base_map_dir: Path = args.base_map_dir.resolve()
    sensors_to_load: set[str] = set([s.strip() for s in args.sensors.split(",") if s.strip()])

    #visualizator creation
    pr_vis = PR_results_visual(db_map_dir, 
                               q_map_dir, 
                               #base_map_dir, 
                               sensors_to_load=sensors_to_load, 
                               transform_matrix_path=args.transform_matrix, 
                               db_feature_map_dir=db_feature_map_dir,
                               q_feature_map_dir=q_feature_map_dir)

    # query-database matching results counting
    results_dir: Path = args.results_dir.resolve()
    if not results_dir.exists():
        raise FileNotFoundError(f"results-path directory does not exist: {results_dir}")
    ns = []
    for i, path in enumerate(os.listdir(results_dir)):
            ns.append(int(path[7:11]))
    ns = sorted(ns)

    #visualization pipeline 
    with write_rerun_visualization(args.output, args.app_id):

        #visualize trajectory in all time sets
        pr_vis.q_map_vis.visualize_trajectory(Path("base_trajectory"))

        #visualize base map in all time sets
        #pr_vis.draw_base_pcd_map(path=Path("base_map"))

        #for all query-results matching results
        for i, n in enumerate(ns):

            #load results
            res = LocalizationResult.load(results_dir / f"result_{n:04d}.json")

            #set time for matching visualization and visualize match
            rr.set_time("time", timestamp=i)
            pr_vis.visualize_PR_result(n, res, path=Path("q"))

if __name__ == "__main__":
    main()