import numpy as np
import yaml
from pathlib import Path
from tf_math import Transform as Tf


DEFAULT_CONFIG_PATH = (
    "/home/docker_mmpr/multimodal-place-recognition/src/mmpr/configs/default.yaml"
)


class RANSACOutlierRejector:
    """
    A wrapper class for estimating a robust average transformation using RANSAC.

    This class reads parameters from a YAML config file and applies a RANSAC-like
    procedure to select a consensus transformation from noisy transformation estimates.

    Attributes
    ----------
    n_samples : int
        Number of transformations to randomly sample in each iteration.
    thr_deg : float
        Rotation threshold in degrees for considering an inlier.
    thr_meters : float
        Translation threshold in meters for considering an inlier.
    max_iterations : int
        Maximum number of RANSAC iterations.
    n_inliers : int
        Minimum number of inliers required to update the best model.
    """

    def __init__(self, config_path: str | Path = DEFAULT_CONFIG_PATH):
        """
        Initialize the estimator by reading configuration parameters.

        Parameters
        ----------
        config_path : str or Path
            Path to the YAML configuration file.
        """
        self.load_config(config_path)

    def load_config(self, config_path: str | Path):
        """Load RANSAC parameters from a YAML configuration file."""
        with open(config_path, "r") as f:
            cfg = yaml.safe_load(f)
        cfg = cfg.get("outlier_rejector", {})
        self.n_samples = cfg.get("n_samples", 10)
        self.thr_deg = cfg.get("thr_deg", 20.0)
        self.thr_meters = cfg.get("thr_meters", 4.0)
        self.max_iterations = cfg.get("max_iterations", 100)
        self.n_inliers = cfg.get("n_inliers", 1)

    def estimate(self, transforms: Tf):
        """
        Estimate the best transformation using RANSAC.

        Parameters
        ----------
        transforms : list of tf_math Transformations

        Returns
        -------
        best_transform : tf_math Transform object
            The estimated consensus transformation (or None if not found).
        best_inlier_ids : np.ndarray
            Indices of inlier transformations.
        best_outlier_ids : np.ndarray
            Indices of outlier transformations.
        """
        best_transform = None
        best_inlier_ids = []
        best_outlier_ids = []

        for _ in range(self.max_iterations):
            subset_ids = np.asarray(
                sorted(np.random.choice(len(transforms), self.n_samples, replace=False))
            )
            subset_avg_tf = transforms[subset_ids].mean()

            # Compute relative distances
            distance_6d = subset_avg_tf.inv() * transforms[subset_ids]
            rot_distances_deg = distance_6d.rotation.magnitude() * 180 / np.pi
            tra_distances = np.linalg.norm(distance_6d.translation, axis=1)

            subset_inlier_ids = np.where(
                (rot_distances_deg < self.thr_deg) & (tra_distances < self.thr_meters)
            )[0]

            inlier_ids = subset_ids[subset_inlier_ids]
            outlier_ids = np.asarray(
                list(set(np.arange(len(transforms))) - set(inlier_ids))
            )

            if (
                len(inlier_ids) > len(best_inlier_ids)
                and len(inlier_ids) > self.n_inliers
            ):
                best_transform = transforms[inlier_ids].mean()
                best_inlier_ids = inlier_ids
                best_outlier_ids = outlier_ids

        return best_transform, best_inlier_ids, best_outlier_ids
