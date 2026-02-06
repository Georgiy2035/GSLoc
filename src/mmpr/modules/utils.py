import numpy as np
import pandas as pd
import open3d as o3d
from pathlib import Path
import matplotlib.pyplot as plt
from tf_math import Transform
import pypcd4 as pypcd
from icp.common.structures import Lidar, RegistrationResult

def rotation_error(R: np.ndarray, degrees: bool = True) -> float:
    """Return rotation error (angle) from a rotation matrix.

    Args:
        R: Rotation Matrix of the relative transformation.
        degrees: if return in degrees or radians

    Returns:
        Rotation error (angle) in degrees or radians
    """
    R = np.asarray(R, dtype=float)[:3, :3]
    cos_theta = (np.trace(R) - 1.0) / 2.0
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    angle = np.arccos(cos_theta)
    return float(np.degrees(angle) if degrees else angle)


def compute_candidate_errors(query_pose, candidate_db_pose, candidate_estimated_pose):
    """Compute PR and Reg translation/rotation errors for one candidate.

    Args:
        query_pose:
        candidate_db_pose:
        candidate_estimated_pose:

    Returns:
        Dictionary of errors for PR and Reg
    """

    query_tf = Transform.from_quat(quat=query_pose[3:], translation=query_pose[:3])
    db_tf = Transform.from_quat(
        quat=candidate_db_pose[3:], translation=candidate_db_pose[:3]
    )
    est_tf = Transform.from_quat(
        quat=candidate_estimated_pose[3:], translation=candidate_estimated_pose[:3]
    )

    pr_err_tf = query_tf.inv() * db_tf
    reg_err_tf = query_tf.inv() * est_tf

    return {
        "pr_translation_error": np.linalg.norm(pr_err_tf.translation),
        "pr_rotation_error": rotation_error(pr_err_tf.as_transformation_matrix()),
        "reg_translation_error": np.linalg.norm(reg_err_tf.translation),
        "reg_rotation_error": rotation_error(reg_err_tf.as_transformation_matrix()),
    }


def aggregate_errors(errors_list):
    """Aggregate errors over multiple queries.

    Args:
        errors_list: List of errors for each query

    Returns:
        Dictionary of errors for PR and Reg
    """
    errors = {k: [] for k in errors_list[0].keys()}
    for e in errors_list:
        for k, v in e.items():
            errors[k].append(v)
    return {k: np.array(v) for k, v in errors.items()}


def plot_xy_positions(positions: np.ndarray, title: str = "2D Positions (x,y)"):
    plt.figure(figsize=(8, 6))
    plt.scatter(positions[:, 0], positions[:, 1], s=8, alpha=0.7)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(title)
    plt.axis("equal")
    plt.grid(True)
    plt.show()


def plot_query_vs_db(query_xy: np.ndarray, db_xy: np.ndarray, title: str = ""):
    plt.figure(figsize=(8, 6))
    plt.scatter(db_xy[:, 0], db_xy[:, 1], c="blue", label="Database", s=1, alpha=0.5)
    plt.scatter(query_xy[:, 0], query_xy[:, 1], c="red", label="Query", s=1, alpha=0.7)
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.title(title)
    plt.show()


def summarize_metrics(metrics: dict) -> pd.DataFrame:
    stats = {}
    for k, arr in metrics.items():
        stats[k] = {
            "mean": np.mean(arr),
            "median": np.median(arr),
            "std": np.std(arr),
            "min": np.min(arr),
            "max": np.max(arr),
            "90th %ile": np.percentile(arr, 90),
        }
    return pd.DataFrame(stats).T


class SimplePCDLoader:
    def __init__(
        self,
        map_root: Path,
        scans_subdir: str = "scans",
        transform: np.ndarray | Transform | None = None,
    ):
        self._map_root = map_root
        self._scans_subdir = scans_subdir

        poses_xyz_quat = self._load_trajectory_df(self._map_root / "poses.csv")[
            ["x", "y", "z", "qx", "qy", "qz", "qw"]
        ].to_numpy()
        self._poses = Transform.from_quat(
            quat=poses_xyz_quat[:, 3:], translation=poses_xyz_quat[:, :3]
        )
        if transform is not None:
            if isinstance(transform, np.ndarray):
                if transform.shape == (4, 4):
                    tf = Transform.from_transformation_matrix(transform)
                elif transform.shape == (7,):
                    tf = Transform.from_quat(
                        quat=transform[3:], translation=transform[:3]
                    )
                else:
                    raise ValueError(f"Invalid transform shape: {transform.shape}")
            elif isinstance(transform, Transform):
                tf = transform
            else:
                raise ValueError(f"Invalid transform type: {type(transform)}")
            self._poses = tf * self._poses

        self._scan_paths = sorted(
            list(self._map_root.glob(f"{self._scans_subdir}/*.pcd"))
        )
        assert len(self._scan_paths) == len(self._poses)

    def __len__(self) -> int:
        return len(list(self._map_root.glob(f"{self._scans_subdir}/*.pcd")))

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray]:
        scan_path = self._scan_paths[idx]
        quat, translation = self._poses[idx].as_quat()
        pose = np.concatenate([translation, quat])
        scan = self._load_pcd(scan_path)
        return scan, pose

    def _load_pcd(self, scan_path: Path) -> np.ndarray:
        return np.asarray(o3d.io.read_point_cloud(scan_path).points)

    @staticmethod
    def _load_trajectory_df(traj_path, sep=","):
        traj_df = pd.read_table(
            traj_path,
            header=None,
            sep=sep,
            names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
            comment="#",
        )
        return traj_df


def get_transformed_scan(path, tf=None, ds_rate=None):
    """Get a transformed point cloud with ability of downsampling.

    Args:
        - path (str): pcd file path
        - tf (tf_math Transform): transformation needed to be applied.
        - ds_rate (float): downsampling rate
    Returns:
        - numpy array of the transformed downsampled point cloud of shape [3xN]
    """
    pcd = pypcd.PointCloud.from_path(path).numpy()[:, :3]
    if ds_rate is not None:
        pcd = pcd[np.random.choice(pcd.shape[0], size=int(pcd.shape[0] * ds_rate))]
    if tf is not None:
        pcd = tf.apply(pcd)
    return pcd


def build_minimap_from_candidates(
    candidates, selected_ids=None, root_dir=None, ds_rate=None
):
    """Merge multiple candidate DB scans (by their IDs) into a single mini-map point cloud.
    Args:
        candidates: iterable of candidate objects (from localization_result.candidates)
                    each with fields: idx, db_pose (x,y,z,qx,qy,qz,qw), db_pointcloud_path.
        selected_ids (list[int]|None): if None -> use all in candidates (e.g., top-K). Otherwise filter by IDs.
        root_dir (Path): root path where candidate.db_pointcloud_path is relative to.
        ds_rate (float|None): optional downsample rate per scan before merge.
    Returns:
        mini_map_pts: (M,3) numpy array of merged points in global frame.
        used_ids: list of IDs actually merged.
    """
    mini_parts = []
    used_ids = []
    for c in candidates:
        if (selected_ids is not None) and (c.idx not in selected_ids):
            continue
        # Convert pose
        pose = np.asarray(c.db_pose, dtype=float)
        tf = Transform.from_quat(pose[3:], pose[:3])
        # Read scan path (may be relative under keyframe_map/scans/...)
        scan_path = Path(root_dir) / c.db_pointcloud_path
        pts = get_transformed_scan(scan_path, tf=tf, ds_rate=ds_rate)
        mini_parts.append(pts)
        used_ids.append(c.idx)
    if not mini_parts:
        raise ValueError("No candidates matched selected_ids; minimap would be empty.")
    mini_map_pts = np.vstack(mini_parts)
    return mini_map_pts, used_ids


def align_query(engine, query_pts_init, target_points):
    """Refine query alignment against the mini-map using local registration (ICP/GICP).
    Args:
        query_pts_init (np.ndarray): (M,3) query points in global frame
        target_points (np.ndarray): (M,3) mini-map points in global frame
    Returns:
        correction (Tf), result dict with diagnostics
    """
    pad_target = np.zeros((target_points.shape[0], 2), dtype=np.float32)
    pad_source = np.zeros((query_pts_init.shape[0], 2), dtype=np.float32)
    tgt = Lidar(points=np.hstack([target_points, pad_target]), timestamp=0.0)
    src = Lidar(points=np.hstack([query_pts_init, pad_source]), timestamp=0.0)

    result = engine.align_one(src, tgt)

    return result

def print_pose_refinement_metadata(result: RegistrationResult, max_len: int=20) -> None:
    print("Registration Result:")
    keys = result.__dict__.keys()
    longest_name_len = len(max(list(keys), key=len))
    for key in keys:
        name_length = len(key)
        spaces = ' ' * (longest_name_len - name_length)
        result_attribute = result.__dict__[key]
        if len(repr(result_attribute)) > max_len:
            result_attribute = '[long attribute, not printed]'
        print(f"  {key}{spaces} {result_attribute}")
