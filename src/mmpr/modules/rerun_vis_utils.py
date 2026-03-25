import rerun as rr
import numpy as np
from pathlib import Path
from mmpr.modules.vis_utils import add_feature_map, add_frame, quaternion_angle
from mmpr.data.pcd import SimplePCDLoader
from mmpr.data.camera import SimpleCameraLoader
from mmpr.inference.data import LocalizationResult

BASE_CMAP_5 = [
    [215,25,28],
    [253,174,97],
    [255,255,191],
    [166,217,106],
    [26,150,65],
]

def visualize_camera_pose(
    pose,
    fov_x_deg: float = 60,
    fov_y_deg: float = 40,
    depth: float = 3.0,
    fov_color: list[float] = [255, 200, 0],
    up_strip_color: list[float] = [255, 200, 0],
    point_color: list[float] = [255, 0, 0],
    path: Path = Path("camera/pose")
) -> None:
    
    """Camera FOV visualization for rerun scene
    
    Args:
        pose: list of floats [px, py, pz, qx, qy, qz, qw] camera world quaternion pose
        fov_x_deg: float horizontal FOV size in degrees
        fov_y_deg: float vertical FOV size in degrees
        depth: float FOV depth in meters
        fov_color: list of floats [R, G, B] that represents FOV color
        up_strip_color: list of floats [R, G, B] that represents up strip color of camera frame
        point_color: list of floats [R, G, B] that represents lidar cloud's points color
        path: Path to camera pose in Rerun
    """
    
    translation = pose[:3]
    quaternion = pose[3:]
    
    # FOV corners calculation
    half_x = depth * np.tan(np.radians(fov_x_deg) / 2)
    half_y = depth * np.tan(np.radians(fov_y_deg) / 2)
    
    # Pyramid vertices in local coordinates
    corners_local = np.array([
        [0, 0, 0],           # camera center
        [depth, -half_x, -half_y],  # left bottom
        [depth, half_x, -half_y],   # right bottom
        [depth, half_x, half_y],    # right up
        [depth, -half_x, half_y],   # left up
    ])
    
    # Coordinate transformation to use local coordinates inside camera_name scope
    rr.log(
        str("world" / path),
        rr.Transform3D(
            translation=translation,
            rotation=rr.Quaternion(xyzw=quaternion)
        )
    )
    
    # FOV pyramid logging
    fov_strips = [
        corners_local[1:4], [corners_local[4], corners_local[1]],  # Rectangle base
        [corners_local[0], corners_local[1]],  # Edges
        [corners_local[0], corners_local[2]],
        [corners_local[0], corners_local[3]],
        [corners_local[0], corners_local[4]],
    ]

    up_strip = [corners_local[3:5]]  
    
    rr.log(
        str("world" / path / Path("fov")),
        rr.LineStrips3D(
            strips=fov_strips,
            colors=[*fov_color, 240],  # Transperency adding
        )
    )

    rr.log(str("world" / path / Path("fov_up_strip")), rr.LineStrips3D(strips=up_strip, colors=up_strip_color))
    
    # Camera point
    rr.log(
        str("world" / path / Path("body")),
        rr.Points3D(
            positions=[0, 0, 0],
            colors=point_color,
            radii=0.1,
        )
    )
    
    # View direction
    look_dir = np.array([1, 0, 0])  # X axe in local coordinates
    look_end = look_dir * max(depth * 0.7, 0.7)
    
    rr.log(
        str("world" / path / Path("look_direction")),
        rr.LineStrips3D(
            strips=[[[0, 0, 0], look_end]],
            colors=point_color,
        )
    )

# def visualize_frame(idx: int, 
#                     sensors_to_load: list[str],
#                     path: Path = Path("frame"), 
#                     lidar_color: list[float] = [200, 200, 0], 
#                     fov_color=[255, 0, 255], 
#                     point_color=[255, 0, 0], 
#                     size_lidar: float = 0.01,
#                     static: bool = False) -> None:
#         """
#         The function that visualize 
        
#         idx: id of frame that need to be visualized
#         path: path in the Rerun that will be used to collect frame data
#         """

#         if "lidar" in sensors_to_load:
#             coords, pose, scan_path = self.pcd_loader[idx]
#             rr.log(str("world" / path / Path("lidar")), rr.Transform3D(translation=pose[:3], rotation=rr.Quaternion(xyzw=pose[3:])), static=static)
#             rr.log(str("world" / path / Path("lidar")), rr.Points3D(coords[coords[:, 2] < 1.3], colors=np.array(lidar_color).astype(np.uint8), radii=size_lidar), static=static)

#         # Taking images from all mentioned cameras from camera_loader ang logging it
#         if self.cameras:
#             images, pose, images_paths, feature_map = self.camera_loader[idx]
#         for cam in self.cameras:
#             path_cam = path / Path(cam)
#             rr.log(str("world" / path_cam / "Image"), rr.Image(add_feature_map(add_frame(images[cam], lidar_color), feature_map)))
#             if "fish-eye" in cam:
#                 visualize_camera_pose(pose=pose, 
#                                       fov_x_deg=140, 
#                                       fov_y_deg=120, 
#                                       depth=0.5, 
#                                       fov_color=fov_color, 
#                                       up_strip_color=lidar_color, 
#                                       point_color=point_color, 
#                                       path=path_cam / "Pose")
#             else:
#                 visualize_camera_pose(pose=pose, 
#                                       fov_x_deg=80, 
#                                       fov_y_deg=60, 
#                                       depth=1.0, 
#                                       fov_color=fov_color, 
#                                       up_strip_color=lidar_color, 
#                                       point_color=point_color, 
#                                       path=path_cam / "Pose")

class Map_visual:
    """
    Class for visualization of frames from 1 map in IndoorMMPR dataset
    """

    def __init__(self, map_dir: Path, 
                 sensors_to_load: list[str] = ["lidar", "cam_fish-eye_left"], 
                 transform_matrix_path: Path = "transform_to_map1.npy",
                 feature_map_dir: Path = "/mnt/external_usb_hdd/6YL/Datasets/SberRobotics/mmpr_pca/map1_pca/pca") -> None:
        """
        Docstring for __init__

        map_dir: path to map from which the data will be visualized
        sensors_to_load: list of sensors that will be used in visualization
        transform_matrix_path: coordinate transformation matrix path
        """
        if not map_dir.exists():
            raise FileNotFoundError(f"map_dir does not exist: {map_dir}")
        self.map_dir = map_dir

        if (map_dir / transform_matrix_path).exists():
            transform_matrix = np.load(map_dir / transform_matrix_path)
        elif transform_matrix_path.exists():
            transform_matrix = np.load(transform_matrix_path)
        else:
            transform_matrix = None

        self.sensors_to_load = sensors_to_load
        self.cameras = list(filter(lambda x: "cam" in x, sensors_to_load))

        # Init data loaders
        self.pcd_loader = SimplePCDLoader(map_dir, scans_subdir="scans", T_map_to_world=transform_matrix)
        self.camera_loader = SimpleCameraLoader(map_dir, self.cameras, T_map_to_world=transform_matrix, feature_map_dir=feature_map_dir)


    def visualize_frame(self, 
                        idx: int, 
                        path: Path = Path("frame"), 
                        lidar_color: list[float] = [200, 200, 0], 
                        fov_color=[255, 0, 255], 
                        point_color=[255, 0, 0], 
                        size_lidar: float = 0.01,
                        static: bool = False) -> None:
        """
        The function that visualize 
        
        idx: id of frame that need to be visualized
        path: path in the Rerun that will be used to collect frame data
        """

        if "lidar" in self.sensors_to_load:
            coords, pose, scan_path = self.pcd_loader[idx]
            rr.log(str("world" / path / Path("lidar")), rr.Transform3D(translation=pose[:3], rotation=rr.Quaternion(xyzw=pose[3:])), static=static)
            rr.log(str("world" / path / Path("lidar")), rr.Points3D(coords[coords[:, 2] < 1.3], colors=np.array(lidar_color).astype(np.uint8), radii=size_lidar), static=static)

        # Taking images from all mentioned cameras from camera_loader ang logging it
        if self.cameras:
            images, pose, images_paths, feature_map = self.camera_loader[idx]
        for cam in self.cameras:
            path_cam = path / Path(cam)
            rr.log(str("world" / path_cam / "Image"), rr.Image(add_feature_map(add_frame(images[cam], lidar_color), feature_map)))
            if "fish-eye" in cam:
                visualize_camera_pose(pose=pose, 
                                      fov_x_deg=140, 
                                      fov_y_deg=120, 
                                      depth=0.5, 
                                      fov_color=fov_color, 
                                      up_strip_color=lidar_color, 
                                      point_color=point_color, 
                                      path=path_cam / "Pose")
            else:
                visualize_camera_pose(pose=pose, 
                                      fov_x_deg=80, 
                                      fov_y_deg=60, 
                                      depth=1.0, 
                                      fov_color=fov_color, 
                                      up_strip_color=lidar_color, 
                                      point_color=point_color, 
                                      path=path_cam / "Pose")


    def visualize_map(self, stride: int = 1, one_shot: bool = False, path: Path = Path("all_map"), color: list[float] = [0, 0, 0], size: float = 0.007) -> None:
        if not one_shot:
            for i in range(0, len(self.pcd_loader._poses), stride):
                rr.set_time("time", timestamp=i)
                self.visualize_frame(i, lidar_color=color, size_lidar=size)
        else: 
            for i in range(0, len(self.pcd_loader._poses), stride):
                self.visualize_frame(i, path=path / Path(f"frame_{i}"), lidar_color=color, size_lidar=size, static=True)


    def visualize_trajectory(self, path: Path, color: list[float] = [255, 255, 0]) -> None:
        traj_xyz = self.pcd_loader._poses[:, :3].astype(np.float32)
        rr.log(str("world" / path / "trajectory"), rr.LineStrips3D(traj_xyz.reshape(1, -1, 3), colors=color, radii=0.02), static=True)



class PR_results_visual:
    def __init__(self,
            db_map_dir: Path, 
            q_map_dir: Path, 
            base_map_dir: Path | None = None,
            sensors_to_load: list[str] = ["lidar", "cam_fish-eye_left"], 
            transform_matrix_path: Path = "transform_to_map1.npy",
            db_feature_map_dir: Path = "/mnt/external_usb_hdd/6YL/Datasets/SberRobotics/mmpr_pca/map1_pca/pca",
            q_feature_map_dir: Path = "/mnt/external_usb_hdd/6YL/Datasets/SberRobotics/mmpr_pca/map2_pca/pinkin_ek/data/sber/pca",
            ):
        
        self.db_map_vis = Map_visual(db_map_dir, sensors_to_load, transform_matrix_path, db_feature_map_dir)
        self.q_map_vis = Map_visual(q_map_dir, sensors_to_load, transform_matrix_path, q_feature_map_dir)
        if base_map_dir is not None:
            self.base_map_vis = Map_visual(base_map_dir, ['lidar'], transform_matrix_path)
    
    def visualize_PR_result(self, qid, result: LocalizationResult, k: int = 5, path: Path = Path("pr_scene"), q_color: list[float] = [0, 255, 255], db_color: list[float] = [50, 50, 50]):
        self.q_map_vis.visualize_frame(qid, 
                                       path / Path("query_frame"), 
                                       lidar_color=q_color, 
                                       point_color=[255, 0, 0])

        for i in range(k):
            lidar_color = BASE_CMAP_5[-i - 1]
            self.db_map_vis.visualize_frame(result.candidates[i].idx, 
                                            path / Path(f"candidate{i + 1}_frame"), 
                                            lidar_color=lidar_color, 
                                            point_color=[0, 0, 255], 
                                            fov_color=[148, 0, 211])
            #rr.log(str("world" / path / f"candidate{i + 1}_frame"), rr.Scalars(result.candidates[i].pr_distance))

        ate = np.linalg.norm(result.candidates[0].estimated_pose[:3] - self.q_map_vis.pcd_loader._poses[qid][:3])
        are = quaternion_angle(result.candidates[0].estimated_pose[3:] , self.q_map_vis.pcd_loader._poses[qid][3:])
        rr.log(str("world" / path / f"stats"), rr.TextDocument(f"ATE: {ate:.2f} meters\nARE: {are:.2f} degrees"))
        rr.log(str("world" / path / f"description"), rr.TextDocument(f"Top 5 candidates for query (blue) from database (from green — 1st to red — 5th)"))

    def draw_base_pcd_map(self, path: Path = Path("base_map"), color: list[float] = [0, 0, 0], size: float = 0.01):
        self.base_map_vis.visualize_map(stride=1, one_shot=True, path=path, color=color, size=size)

    # def visualize_PR_result_onboard(self, pcd, pose7, result: LocalizationResult, k: int = 5, path: Path = Path("pr_scene"), q_color: list[float] = [0, 255, 255], db_color: list[float] = [50, 50, 50]):
    #     self.q_map_vis.visualize_frame(qid, 
    #                                    path / Path("query_frame"), 
    #                                    lidar_color=q_color, 
    #                                    point_color=[255, 0, 0])

    #     for i in range(k):
    #         lidar_color = BASE_CMAP_5[-i - 1]
    #         self.db_map_vis.visualize_frame(result.candidates[i].idx, 
    #                                         path / Path(f"candidate{i + 1}_frame"), 
    #                                         lidar_color=lidar_color, 
    #                                         point_color=[0, 0, 255], 
    #                                         fov_color=[148, 0, 211])
        
    #     rr.log(str("world" / path / f"description"), rr.TextDocument(f"Top 5 candidates for query (blue) from database (from green — 1st to red — 5th)"))



class write_rerun_visualization(object):
    def __init__(self, output_file: Path, app_id: str = "test_app_id"):
        self.output_file = output_file
        self.app_id = app_id

    def __enter__(self):
        rr.init(self.app_id)
        rr.save(str(self.output_file))

    def __exit__(self, type, value, traceback):
        rr.disconnect()
    
