"""
InferenceIndexBuilder - A wrapper class for using OPR inference indices.

It provides an interface for building FAISS indices for place recognition and localization.
"""

import json
import logging
import hashlib
from pathlib import Path
from typing import Optional, Union, List, Tuple, Dict, Any
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from tqdm import tqdm
import yaml

import open3d as o3d
import MinkowskiEngine as ME

from .utils import compute_candidate_errors, aggregate_errors


# OPR imports
from opr.models.place_recognition import MinkLoc3D
from opr.inference.index import FaissFlatIndex
from opr.inference.pipelines import (
    PlaceRecognitionPipeline,
    LocalizationPipeline,
    RansacPointCloudRegistrationPipeline,
)


@dataclass
class IndexConfig:
    """Configs for index building."""

    dist_thresh: float = 0.5
    angle_thresh_deg: float = 30
    pointcloud_quantization_size: float = 0.05
    model_path: Optional[str] = None
    device: str = "cuda"
    start_with_first: bool = True

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "IndexConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get("index_config", {}))

    def to_yaml(self, yaml_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_dict = {"index_config": asdict(self)}
        with open(yaml_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)

    def get_hash(self) -> str:
        """Get a hash of the configuration for caching purposes."""
        # Create a deterministic string representation
        config_str = json.dumps(asdict(self), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]


@dataclass
class LocalizationConfig:
    """Configs for localization."""

    k: int = 5  # number of candidates
    voxel_downsample_size: float = 0.5

    @classmethod
    def from_yaml(cls, yaml_path: Union[str, Path]) -> "LocalizationConfig":
        """Load configuration from YAML file."""
        with open(yaml_path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict.get("localization_config", {}))


class InferenceIndexBuilder:
    """
    A wrapper class for using OPR inference indices.

    Methods:
    1. Build FAISS indices from trajectory data and point clouds.
    2. Filter poses based on spatial and angular thresholds.
    3. Run place recognition and localization pipelines.
    4. Cache embeddings.
    """

    def __init__(self, config: Optional[Union[IndexConfig, str, Path]] = None):
        """
        Initialize the InferenceIndexBuilder.

        Args:
            config: Configuration for the index building. Can be:
                   - IndexConfig object
                   - Path to YAML configuration file
                   - None (uses default values)
        """
        if isinstance(config, (str, Path)):
            self.config = IndexConfig.from_yaml(config)
        elif isinstance(config, IndexConfig):
            self.config = config
        else:
            self.config = IndexConfig()

        self.device = torch.device(
            self.config.device if torch.cuda.is_available() else "cpu"
        )
        self.logger = self._setup_logger()
        self._load_model()

    def _setup_logger(self) -> logging.Logger:
        """Setup logging for the class."""
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

    def _load_model(self, model_path: Optional[str] = None) -> None:
        """
        Load the MinkLoc3D model.

        Args:
            model_path: Path to model weights.
        """
        self.model = MinkLoc3D()
        model_path = model_path if model_path is not None else self.config.model_path
        assert model_path is not None, "No model path provided."
        self.logger.info(f"Loading model from {model_path}")
        self.model.load_state_dict(torch.load(model_path), strict=True)
        self.model.eval()
        self.model.to(self.device)
        self.logger.info(f"Model loaded on device: {self.device}")

    def load_trajectory_df(
        self, traj_path: Union[str, Path], sep: str = ","
    ) -> pd.DataFrame:
        """
        Load trajectory data from CSV file.

        Args:
            traj_path: Path to trajectory CSV file.
            sep: Column separator

        Returns:
            A DataFrame containing the trajectory data.
        """
        traj_df = pd.read_table(
            traj_path,
            header=None,
            sep=sep,
            names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
            comment="#",
        )
        self.logger.info(f"Loaded frames with {len(traj_df)} entries from {traj_path}")
        return traj_df

    def filter_poses_by_thresholds(
        self,
        df: pd.DataFrame,
        dist_thresh: Optional[float] = None,
        angle_thresh_deg: Optional[float] = None,
        position_columns: List[str] = None,
        quaternion_columns: List[str] = None,
    ) -> pd.DataFrame:
        """
        Filter poses based on spatial and angular thresholds.

        Args:
            df: DataFrame with at least the position and quaternion columns.
            dist_thresh: Minimum Euclidean distance (meters) to accept.
            angle_thresh_deg: Angular distance threshold in degrees when spatially close.
            position_columns: Names of position columns in order [x, y, z].
            quaternion_columns: Names of quaternion columns in order [qx, qy, qz, qw].

        Returns:
            Filtered DataFrame.
        """
        dist_thresh = (
            dist_thresh if dist_thresh is not None else self.config.dist_thresh
        )
        angle_thresh_deg = (
            angle_thresh_deg
            if angle_thresh_deg is not None
            else self.config.angle_thresh_deg
        )
        position_columns = (
            position_columns if position_columns is not None else ["x", "y", "z"]
        )
        quaternion_columns = (
            quaternion_columns
            if quaternion_columns is not None
            else ["qx", "qy", "qz", "qw"]
        )

        required_columns = position_columns + quaternion_columns
        missing_columns = [c for c in required_columns if c not in df.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        if len(df) == 0:
            return df.copy()

        positions = df[position_columns].to_numpy(dtype=float, copy=True)
        quaternions = df[quaternion_columns].to_numpy(dtype=float, copy=True)
        selected_indices: List[int] = []

        def _normalize_quaternion(q: np.ndarray) -> np.ndarray:
            """Return unit quaternion; if zero-norm, return input."""
            norm = np.linalg.norm(q)
            return q / norm if norm > 0 else q

        def _quaternion_angle_diff_deg(q1: np.ndarray, q2: np.ndarray) -> float:
            """Compute minimal angular difference between two quaternions in degrees."""
            q1n = _normalize_quaternion(q1)
            q2n = _normalize_quaternion(q2)
            dot = float(np.clip(np.abs(np.dot(q1n, q2n)), -1.0, 1.0))
            angle_rad = 2.0 * np.arccos(dot)
            return float(np.degrees(angle_rad))

        if self.config.start_with_first:
            selected_indices.append(0)

        selected_positions = (
            positions[selected_indices]
            if selected_indices
            else np.empty((0, positions.shape[1]), dtype=float)
        )
        selected_quaternions = (
            quaternions[selected_indices]
            if selected_indices
            else np.empty((0, quaternions.shape[1]), dtype=float)
        )

        start_i = 1 if self.config.start_with_first else 0
        for i in range(start_i, len(df)):
            pos_i = positions[i]
            quat_i = quaternions[i]

            if selected_positions.shape[0] == 0:
                selected_indices.append(i)
                selected_positions = np.vstack([selected_positions, pos_i])
                selected_quaternions = np.vstack([selected_quaternions, quat_i])
                continue

            dists = np.linalg.norm(selected_positions - pos_i, axis=1)
            nearest_sel_idx = int(np.argmin(dists))
            min_dist = float(dists[nearest_sel_idx])

            if min_dist > dist_thresh:
                should_select = True
            else:
                ang_deg = _quaternion_angle_diff_deg(
                    quat_i, selected_quaternions[nearest_sel_idx]
                )
                should_select = ang_deg > angle_thresh_deg

            if should_select:
                selected_indices.append(i)
                selected_positions = np.vstack([selected_positions, pos_i])
                selected_quaternions = np.vstack([selected_quaternions, quat_i])

        filtered_df = df.iloc[selected_indices]
        self.logger.info(
            f"Selected {len(filtered_df)} / {len(df)} frames "
            f"({len(filtered_df) / max(1, len(df)):.1%})"
        )
        return filtered_df

    def _get_cache_paths(self, output_dir: Path, config_hash: str) -> Dict[str, Path]:
        """Get paths for cached files."""
        cache_dir = output_dir / "cache"
        cache_dir.mkdir(exist_ok=True)

        return {
            "embeddings": cache_dir / f"embeddings_{config_hash}.npy",
            "filtered_indices": cache_dir / f"filtered_indices_{config_hash}.npy",
            "config": cache_dir / f"config_{config_hash}.yaml",
        }

    def _save_cache(
        self,
        output_dir: Path,
        config_hash: str,
        embeddings: np.ndarray,
        filtered_indices: np.ndarray,
    ) -> None:
        """Save embeddings and indices to cache."""
        cache_paths = self._get_cache_paths(output_dir, config_hash)

        # Save embeddings and indices
        np.save(cache_paths["embeddings"], embeddings)
        np.save(cache_paths["filtered_indices"], filtered_indices)

        # Save config for reference
        self.config.to_yaml(cache_paths["config"])

        self.logger.info(f"Cached embeddings and indices with hash: {config_hash}")

    def _load_cache(
        self, output_dir: Path, config_hash: str
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """Load embeddings and indices from cache if available."""
        cache_paths = self._get_cache_paths(output_dir, config_hash)

        if all(
            path.exists()
            for path in [cache_paths["embeddings"], cache_paths["filtered_indices"]]
        ):
            embeddings = np.load(cache_paths["embeddings"])
            filtered_indices = np.load(cache_paths["filtered_indices"])

            self.logger.info(
                f"Loaded cached embeddings and indices with hash: {config_hash}"
            )
            return embeddings, filtered_indices

        return None

    def read_scan(
        self, scan_filepath: Union[str, Path]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Read a point cloud scan from a file.

        Args:
            scan_filepath: Path to the point cloud file

        Returns:
            Tuple of (coordinates, features)
        """
        scan = o3d.io.read_point_cloud(str(scan_filepath))
        if not scan.has_points():
            raise ValueError(f"Scan file {scan_filepath} is empty or invalid.")

        scan = np.asarray(scan.points)
        coordinates = scan[:, :3]  # Get the first three columns (x, y, z)

        if scan.shape[1] == 3:
            features = np.ones((coordinates.shape[0], 1))
        elif scan.shape[1] == 4:
            features = scan[:, 3:4]  # Get the fourth column (intensity)
        else:
            raise ValueError(
                f"Unexpected scan format with shape {scan.shape}. Expected 3 or 4 columns."
            )

        return coordinates, features

    def to_batch(self, coords: np.ndarray, feats: np.ndarray) -> Dict[str, Tensor]:
        """
        Convert coordinates and features to MinkowskiEngine batch format.

        Args:
            coords: Point cloud coordinates
            feats: Point cloud features

        Returns:
            Dictionary with batched coordinates and features
        """
        coords_t = torch.from_numpy(coords).float()
        feats_t = torch.from_numpy(feats).float()

        quantized_coords, quantized_feats = ME.utils.sparse_quantize(
            coordinates=coords_t,
            features=feats_t,
            quantization_size=self.config.pointcloud_quantization_size,
        )

        return {
            "pointclouds_lidar_coords": ME.utils.batched_coordinates(
                [quantized_coords]
            ),
            "pointclouds_lidar_feats": torch.cat([quantized_feats]),
        }

    def compute_descriptors(
        self, scan_paths: List[Union[str, Path]], base_dir: Union[str, Path]
    ) -> np.ndarray:
        """
        Compute descriptors for a list of scan files.

        Args:
            scan_paths: List of scan file paths (relative to base_dir)
            base_dir: Base directory containing the scans

        Returns:
            Array of descriptors
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        base_dir = Path(base_dir)
        descriptors = []

        self.logger.info(f"Computing descriptors for {len(scan_paths)} scans...")

        for i, scan_path in enumerate(tqdm(scan_paths, desc="Processing scans")):
            coords, feats = self.read_scan(base_dir / scan_path)
            batch = self.to_batch(coords, feats)
            batch = {k: v.to(self.device) for k, v in batch.items()}

            with torch.no_grad():
                desc = self.model(batch)

            descriptors.append(desc["final_descriptor"].cpu().numpy())

        descriptors = np.concatenate(descriptors, axis=0)
        self.logger.info(f"Computed descriptors with shape: {descriptors.shape}")

        return descriptors

    def build_index(
        self,
        map_dir: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None,
        poses_file: str = "poses.csv",
        use_cache: bool = True,
    ) -> FaissFlatIndex:
        """
        Build a complete FAISS index from trajectory and scan data.

        Args:
            map_dir: Directory containing poses.csv and scans/ subdirectory
            output_dir: Output directory for index files (defaults to map_dir)
            poses_file: Name of the poses file
            use_cache: Whether to use cached embeddings if available

        Returns:
            Built FaissFlatIndex
        """
        map_dir = Path(map_dir)
        output_dir = Path(output_dir) if output_dir else map_dir

        # Load and filter trajectory
        traj_df = self.load_trajectory_df(map_dir / poses_file)
        filtered_df = self.filter_poses_by_thresholds(traj_df)

        # Get configuration hash for caching
        config_hash = self.config.get_hash()
        filtered_indices = filtered_df.index.to_numpy()

        # Try to load from cache first
        cached_data = None
        if use_cache:
            cached_data = self._load_cache(output_dir, config_hash)

        if cached_data is not None:
            descriptors, cached_filtered_indices = cached_data

            # Verify that cached indices match current filtering
            if np.array_equal(filtered_indices, cached_filtered_indices):
                self.logger.info("Using cached embeddings")
            else:
                self.logger.warning(
                    "Cached indices don't match current filtering, recomputing..."
                )
                cached_data = None

        if cached_data is None:
            # Get scan paths
            scans_dir = map_dir / "scans"
            all_scans = sorted(scans_dir.glob("*.pcd"))
            selected_indices = filtered_df.index.tolist()
            selected_scans = [all_scans[i] for i in selected_indices]
            selected_scan_paths = [
                str(scan.relative_to(map_dir)) for scan in selected_scans
            ]

            self.logger.info(
                f"Selected {len(selected_scans)} scans from {len(all_scans)} total"
            )

            # Compute descriptors
            descriptors = self.compute_descriptors(selected_scan_paths, map_dir)

            # Cache the results
            if use_cache:
                self._save_cache(output_dir, config_hash, descriptors, filtered_indices)
        else:
            # If using cache, still need scan paths for index building
            scans_dir = map_dir / "scans"
            all_scans = sorted(scans_dir.glob("*.pcd"))
            selected_indices = filtered_df.index.tolist()
            selected_scans = [all_scans[i] for i in selected_indices]
            selected_scan_paths = [
                str(scan.relative_to(map_dir)) for scan in selected_scans
            ]

        # Save descriptors to output directory
        np.save(output_dir / "descriptors.npy", descriptors)
        self.logger.info(f"Saved descriptors to {output_dir / 'descriptors.npy'}")

        # Build index files
        self._build_index_files(
            filtered_df, selected_scan_paths, descriptors, output_dir
        )

        # Load and return the index
        index = FaissFlatIndex.load(output_dir)
        self.logger.info(f"Built index with {index.size()} entries, dim={index.dim()}")

        return index

    def _build_index_files(
        self,
        filtered_df: pd.DataFrame,
        scan_paths: List[str],
        descriptors: np.ndarray,
        output_dir: Path,
    ) -> None:
        """Build the index files (meta.parquet and schema.json)."""
        # Build meta.parquet
        poses = filtered_df[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(
            dtype=float
        )
        poses_list = [list(p) for p in poses]

        meta = pd.DataFrame(
            {
                "idx": filtered_df.index.to_numpy(dtype=np.int64),
                "pose": poses_list,
                "pointcloud_path": scan_paths,
            }
        )
        meta.to_parquet(output_dir / "meta.parquet")

        # Build schema.json
        schema = {
            "version": "1",
            "dim": descriptors.shape[1],
            "metric": "l2",
            "created_at": "",
            "opr_version": "",
        }
        (output_dir / "schema.json").write_text(json.dumps(schema))

    def create_place_recognition_pipeline(
        self, index: FaissFlatIndex
    ) -> PlaceRecognitionPipeline:
        """
        Create a place recognition pipeline.

        Args:
            index: The FAISS index to use

        Returns:
            Configured PlaceRecognitionPipeline
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")

        return PlaceRecognitionPipeline(
            index=index, model=self.model, device=str(self.device)
        )

    def evaluate_localization(
        self,
        pipeline: LocalizationPipeline,
        scans_loader,
        k: int = 5,
        save_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Run localization on all scans in a loader and compute error metrics.
        """
        results = []
        for query_pc, query_pose in tqdm(scans_loader, desc="Evaluating"):
            query_pc_dict = self.to_batch(query_pc, np.ones((len(query_pc), 1)))
            query_pc_dict = {k: v.to(self.device) for k, v in query_pc_dict.items()}

            prediction = pipeline.infer(
                pr_input=query_pc_dict, query_pc=torch.from_numpy(query_pc), k=k
            )
            for cand in prediction.candidates:
                errors = compute_candidate_errors(
                    query_pose, cand.db_pose, cand.estimated_pose
                )
                results.append(errors)

        aggregated = aggregate_errors(results)

        if save_path:
            pd.DataFrame(results).to_json(save_path, orient="records", lines=True)

        return aggregated

    @staticmethod
    def get_xy_positions(index: FaissFlatIndex) -> np.ndarray:
        """
        Extract 2D (x, y) positions from a FaissFlatIndex.
        """
        if not hasattr(index, "_db_pose"):
            raise AttributeError("Index does not have stored poses (_db_pose).")
        return index._db_pose[:, :2]

    def create_localization_pipeline(
        self,
        index: FaissFlatIndex,
        index_root: Union[str, Path],
        place_recognition_pipeline: Optional[Any] = None,
        config: Optional[LocalizationConfig] = None,
    ) -> LocalizationPipeline:
        """
        Create a localization pipeline.

        Args:
            index: The FAISS index to use
            index_root: Root directory containing the index data
            place_recognition_pipeline: Optional place recognition model
            config: Localization configuration

        Returns:
            Configured LocalizationPipeline
        """
        config = config or LocalizationConfig()

        if place_recognition_pipeline is None:
            place_recognition_pipeline = self.create_place_recognition_pipeline(index)

        registration_pipeline = RansacPointCloudRegistrationPipeline(
            voxel_downsample_size=config.voxel_downsample_size
        )

        return LocalizationPipeline(
            index=index,
            place_recognition=place_recognition_pipeline,
            registration=registration_pipeline,
            index_root=Path(index_root),
        )

    def infer_place_recognition(
        self,
        pipeline: PlaceRecognitionPipeline,
        query_scan_path: Union[str, Path],
        k: int = 5,
    ) -> Any:
        """
        Run place recognition inference on a query scan.

        Args:
            pipeline: The place recognition pipeline
            query_scan_path: Path to the query scan
            k: Number of nearest neighbors to return

        Returns:
            Place recognition result
        """
        coords, feats = self.read_scan(query_scan_path)
        batch = self.to_batch(coords, feats)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        return pipeline.infer(input_data=batch, k=k)

    def infer_localization(
        self,
        pipeline: LocalizationPipeline,
        query_scan_path: Union[str, Path],
        k: int = 5,
    ) -> Any:
        """
        Run localization inference on a query scan.

        Args:
            pipeline: The localization pipeline
            query_scan_path: Path to the query scan
            k: Number of candidates to consider

        Returns:
            Localization result
        """
        coords, feats = self.read_scan(query_scan_path)
        batch = self.to_batch(coords, feats)
        batch = {k: v.to(self.device) for k, v in batch.items()}

        query_pc = torch.from_numpy(coords).float()

        return pipeline.infer(pr_input=batch, query_pc=query_pc, k=k)
