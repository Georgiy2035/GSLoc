"""Inference pipelines for place recognition, registration, and localization.

This module implements:
- PlaceRecognitionPipeline: Top-k place recognition using FAISS index
- SequencePlaceRecognitionPipeline: Streaming sequence-aware PR with CPF
- RansacPointCloudRegistrationPipeline: Open3D RANSAC-based registration
- LocalizationPipeline: Full localization combining PR + registration

This code is based on the OpenPlaceRecognition library (Apache 2.0 License).
Source: https://github.com/OPR-Project/OpenPlaceRecognition
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Any, Deque, Literal, Dict

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation
from torch import Tensor, nn
# import MinkowskiEngine as ME

from mmpr.inference.data import (
    LocalizationResult,
    LocalizedCandidate,
    PlaceRecognitionResult,
    RegistrationResult,
    SequencePRDebug,
)
from mmpr.inference.index import Index
from mmpr.inference.io import PointCloudStore
from opr.utils import init_model, parse_device


# =============================================================================
# Helper Functions
# =============================================================================


def _candidate_pool_fusion(
    distances: np.ndarray,  # [N, per_k]
    indices: np.ndarray,  # [N, per_k]
    final_k: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Fuse per-frame candidates using Candidate Pool Fusion.

    Args:
        distances: Per-frame raw distances (smaller is better), shape [N, per_k].
        indices: Per-frame internal row ids, shape [N, per_k].
        final_k: Number of final fused candidates to return.

    Returns:
        (fused_distances, fused_indices): Both of shape [final_k]
            (or shorter if not enough uniques).
    """
    if distances.size == 0 or indices.size == 0:
        return np.empty((0,), dtype=np.float32), np.empty((0,), dtype=np.int64)

    flat_d = distances.reshape(-1)
    flat_i = indices.reshape(-1)

    order = np.argsort(flat_d)
    sorted_d = flat_d[order]
    sorted_i = flat_i[order]

    # Deduplicate by first occurrence (best distance)
    unique_i, first_pos = np.unique(sorted_i, return_index=True)
    uniq_d = sorted_d[first_pos]

    # Re-sort by distance because np.unique does not preserve order
    re_order = np.argsort(uniq_d)
    fused_i = unique_i[re_order]
    fused_d = uniq_d[re_order]

    if fused_i.shape[0] > final_k:
        fused_i = fused_i[:final_k]
        fused_d = fused_d[:final_k]

    # Ensure dtypes
    fused_d = fused_d.astype(np.float32, copy=False)
    fused_i = fused_i.astype(np.int64, copy=False)
    return fused_d, fused_i


def _pose7_to_matrix(pose7: np.ndarray) -> np.ndarray:
    """Convert pose [tx,ty,tz,qx,qy,qz,qw] to 4x4 matrix."""
    t = pose7[:3]
    q = pose7[3:]
    R = Rotation.from_quat(q).as_matrix()
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _matrix_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 matrix to pose [tx,ty,tz,qx,qy,qz,qw]."""
    Rm = T[:3, :3]
    t = T[:3, 3]
    q = Rotation.from_matrix(Rm).as_quat()
    return np.concatenate([t, q]).astype(np.float64, copy=False)


# =============================================================================
# Place Recognition Pipeline
# =============================================================================


class PlaceRecognitionPipeline:
    """Minimal top-k Place Recognition pipeline using an `Index`.

    The pipeline assumes that the model returns a dict with key `final_descriptor`.
    It returns raw FAISS distances with corresponding dataset indices and poses.
    """

    def __init__(
        self,
        index: Index,
        model: nn.Module,
        model_weights_path: str | Path | None = None,
        device: str | int | torch.device = "cpu",
    ) -> None:
        """Initialize the pipeline.

        Args:
            index: Loaded `Index` instance.
            model: PyTorch model that outputs `{"final_descriptor": Tensor[B,D]}`.
            model_weights_path: Optional path to weights to load.
            device: Torch device spec.
        """
        self.index = index
        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.model.eval()

    def _preprocess_input(self, input_data: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Preprocess input data."""
        out_dict: Dict[str, Tensor] = {}
        for key in input_data:
            if key.startswith("image_"):
                out_dict[f"images_{key[6:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key.startswith("mask_"):
                out_dict[f"masks_{key[5:]}"] = input_data[key].unsqueeze(0).to(self.device)
            elif key == "pointcloud_lidar_coords":
                quantized_coords, quantized_feats = ME.utils.sparse_quantize(
                    coordinates=input_data["pointcloud_lidar_coords"],
                    features=input_data["pointcloud_lidar_feats"],
                    quantization_size=self._pointcloud_quantization_size,
                )
                out_dict["pointclouds_lidar_coords"] = ME.utils.batched_coordinates([quantized_coords]).to(
                    self.device
                )
                out_dict["pointclouds_lidar_feats"] = quantized_feats.to(self.device)
            elif key == "soc":
                out_dict["soc"] = input_data[key].unsqueeze(0).to(self.device)
        return out_dict

    def _prepare_model_input_batch(self, batch: Dict[str, Any]) -> Dict[str, Tensor]:
        """Create model input dict from a collated batch."""
        model_input: Dict[str, Tensor] = {}
        for key, value in batch.items():
            if not isinstance(value, torch.Tensor):
                continue
            if key.startswith("image_"):
                model_input[f"images_{key[6:]}"] = value.to(self.device, non_blocking=True)
            elif key.startswith("mask_"):
                model_input[f"masks_{key[5:]}"] = value.to(self.device, non_blocking=True)
            elif key == "soc":
                model_input[key] = value.to(self.device, non_blocking=True)
            elif key in {"pointcloud_lidar_coords", "pointcloud_lidar_feats"}:
                raise NotImplementedError(
                    "Batched pointcloud preprocessing requires MinkowskiEngine. "
                    "Use an image-only dataset or implement batched sparse quantization."
                )

        if not model_input:
            raise KeyError(
                "No usable tensor inputs found in batch. Expected at least one tensor key starting with `image_`."
            )
        return model_input

    def _extract_descriptors(self, out: dict[str, Tensor]) -> np.ndarray:
        """Normalize model output into a float32 numpy descriptor batch."""
        if "final_descriptor" not in out:
            raise KeyError("Model output must contain 'final_descriptor'")
        desc_t: Tensor = out["final_descriptor"]
        if desc_t.ndim == 1:
            desc_t = desc_t[None, :]
        elif desc_t.ndim != 2:
            raise ValueError("Expected descriptor tensor of shape [D] or [B,D]")
        return desc_t.detach().cpu().numpy().astype(np.float32, copy=False)

    def _search_descriptors(self, descriptors: np.ndarray, k: int) -> list[PlaceRecognitionResult]:
        """Run index search for a descriptor batch and map metadata."""
        inds, dists = self.index.search(descriptors, int(k))
        db_idx, db_pose, _db_pc = self.index.get_meta(inds.reshape(-1))
        db_idx = db_idx.reshape(inds.shape)
        db_pose = db_pose.reshape(*inds.shape, -1)

        results: list[PlaceRecognitionResult] = []
        for row in range(descriptors.shape[0]):
            results.append(
                PlaceRecognitionResult(
                    descriptor=descriptors[row],
                    indices=inds[row].astype(np.int64, copy=False),
                    distances=dists[row].astype(np.float32, copy=False),
                    db_idx=db_idx[row].astype(np.int64, copy=False),
                    db_pose=db_pose[row].astype(np.float32, copy=False),
                )
            )
        return results

    @torch.inference_mode()
    def batch_infer_descriptors(self, batch: Dict[str, Any]) -> np.ndarray:
        """Run batched descriptor extraction."""
        model_input = self._prepare_model_input_batch(batch)
        out = self.model(model_input)
        return self._extract_descriptors(out)

    @torch.inference_mode()
    def batch_infer(self, batch: Dict[str, Any], k: int = 5) -> list[PlaceRecognitionResult]:
        """Run batched PR inference and return one result per sample."""
        descriptors = self.batch_infer_descriptors(batch)
        return self._search_descriptors(descriptors, k=k)

    @torch.inference_mode()
    def infer(self, input_data: dict[str, Tensor], k: int = 5) -> PlaceRecognitionResult:
        """Run a single-sample inference and top-k search.

        Args:
            input_data: Dict of tensors expected by the model forward.
            k: Number of neighbors to retrieve.

        Returns:
            PlaceRecognitionResult: descriptor, raw distances and mapped metadata.

        Raises:
            KeyError: If model output does not contain the `final_descriptor` key.
            ValueError: If the produced descriptor has an unexpected shape.
        """
        with torch.no_grad():
            model_input = self._preprocess_input(input_data)
            out = self.model(model_input)
        descriptors = self._extract_descriptors(out)
        if descriptors.shape[0] != 1:
            raise ValueError("Expected a single descriptor for single-sample inference")
        return self._search_descriptors(descriptors, k=k)[0]


# =============================================================================
# Sequence Place Recognition Pipeline
# =============================================================================


class SequencePlaceRecognitionPipeline:
    """Streaming sequence-aware Place Recognition with a single-frame model.

    Maintains a FIFO window of recent frames. Each call processes one frame,
    caches its descriptor and per-frame top-k, then fuses across the window
    using Candidate Pool Fusion. Descriptor is aggregated across the window.
    """

    def __init__(
        self,
        index: Index,
        model: nn.Module,
        model_weights_path: str | Path | None = None,
        device: str | int | torch.device = "cpu",
        max_window: int = 20,
        per_frame_k: int = 10,
        final_k: int = 10,
        descriptor_agg: Literal["mean", "ema", "last"] = "mean",
        ema_decay: float = 0.9,
        recency_weighting: Literal["none", "linear", "exp"] = "none",
    ) -> None:
        """Initialize the streaming sequence pipeline.

        Args:
            index: Loaded `Index` instance that provides search and metadata.
            model: Single-frame PyTorch model that outputs
                `{"final_descriptor": Tensor[B,D]}`.
            model_weights_path: Optional path to model weights.
            device: Torch device specification.
            max_window: Maximum number of recent frames kept in the FIFO window.
            per_frame_k: Top-k to retrieve per frame (cached).
            final_k: Final fused top-k to return after CPF.
            descriptor_agg: Aggregation strategy for descriptors across the window.
            ema_decay: EMA decay used when `descriptor_agg="ema"`.
            recency_weighting: Optional recency weighting policy for CPF distances.
        """
        self.index = index
        self.device = parse_device(device)
        self.model = init_model(model, model_weights_path, self.device)
        self.model.eval()

        self.max_window = int(max_window)
        self.per_frame_k = int(per_frame_k)
        self.final_k = int(final_k)
        self.descriptor_agg = descriptor_agg
        self.ema_decay = float(ema_decay)
        self.recency_weighting = recency_weighting

        self._records: Deque[PlaceRecognitionResult] = deque()
        self._running_sum_descriptor: np.ndarray | None = None
        self._ema_descriptor: np.ndarray | None = None

    def reset(self) -> None:
        """Clear the window and aggregation state."""
        self._records.clear()
        self._running_sum_descriptor = None
        self._ema_descriptor = None

    def start_new_sequence(self, session_id: str | None = None) -> None:
        """Alias for reset; kept for future session-aware extensions."""
        self.reset()

    @torch.inference_mode()
    def infer(
        self,
        input_frame: dict[str, Tensor],
        k: int | None = None,
        return_debug: bool = False,
    ) -> PlaceRecognitionResult | tuple[PlaceRecognitionResult, SequencePRDebug]:
        """Process one frame, update the window, and return fused top-k.

        Args:
            input_frame: Single-frame input dict expected by the model.
            k: Optional override for final_k.
            return_debug: If True, also return a SequencePRDebug instance.

        Returns:
            PlaceRecognitionResult (and optionally SequencePRDebug): fused candidates
            and aggregated descriptor for the current window.

        Raises:
            KeyError: If model output does not contain `final_descriptor`.
            ValueError: If the produced descriptor has an unexpected shape.
        """
        final_k = int(k) if k is not None else self.final_k

        # 1) Forward pass: single-frame descriptor
        out = self.model(input_frame)
        if "final_descriptor" not in out:
            raise KeyError("Model output must contain 'final_descriptor'")
        desc_t: Tensor = out["final_descriptor"]
        if desc_t.ndim == 2 and desc_t.shape[0] == 1:
            desc = desc_t[0].detach().cpu().numpy().astype(np.float32, copy=False)
        elif desc_t.ndim == 1:
            desc = desc_t.detach().cpu().numpy().astype(np.float32, copy=False)
        else:
            raise ValueError("Expected descriptor of shape [D] or [1,D]")

        # 2) Per-frame search
        inds, dists = self.index.search(desc.reshape(1, -1), self.per_frame_k)
        inds = inds[0]
        dists = dists[0].astype(np.float32, copy=False)

        # 3) Push to window (cache without metadata to avoid extra lookups)
        pr_res = PlaceRecognitionResult(
            descriptor=desc,
            indices=inds,
            distances=dists,
            db_idx=None,
            db_pose=None,
        )
        self._push_record(pr_res)

        # 4) Gather per-frame arrays for fusion
        per_i = (
            np.stack([r.indices for r in self._records], axis=0)
            if self._records
            else np.empty((0, 0), dtype=np.int64)
        )
        per_d = (
            np.stack([r.distances for r in self._records], axis=0)
            if self._records
            else np.empty((0, 0), dtype=np.float32)
        )

        # Optional recency weighting (off by default)
        if self.recency_weighting != "none" and per_d.size > 0:
            N = per_d.shape[0]
            ages = np.arange(
                N - 1, -1, -1, dtype=np.float32
            )  # oldest .. newest? We want oldest larger weight
            if self.recency_weighting == "linear":
                weights = 1.0 + ages / max(1, N - 1)
            else:  # exp
                # exponential growth with window length; tune base as needed
                base = 1.25
                weights = base ** (ages / max(1, N - 1))
            per_d = per_d * weights[:, None]

        # 5) Fuse
        fused_d, fused_i = _candidate_pool_fusion(per_d, per_i, final_k)

        # 6) Aggregate descriptor across window
        agg_desc = self._aggregate_descriptor()

        # 7) Map metadata for fused indices only
        db_idx, db_pose, _db_pc = (
            self.index.get_meta(fused_i)
            if fused_i.size > 0
            else (np.empty((0,), dtype=np.int64), np.empty((0, 7), dtype=np.float32), None)
        )

        fused_res = PlaceRecognitionResult(
            descriptor=agg_desc,
            indices=fused_i,
            distances=fused_d,
            db_idx=db_idx,
            db_pose=db_pose,
        )

        if not return_debug:
            return fused_res

        debug = SequencePRDebug(
            per_frame_indices=per_i,
            per_frame_distances=per_d,
            fused_indices=fused_i,
            fused_distances=fused_d,
            window_size=len(self._records),
            descriptor_agg=self.descriptor_agg,
        )
        return fused_res, debug

    # --- internal helpers ---

    def _push_record(self, rec: PlaceRecognitionResult) -> None:
        # update aggregates for the descriptor
        if self.descriptor_agg == "mean":
            if self._running_sum_descriptor is None:
                self._running_sum_descriptor = rec.descriptor.astype(np.float32, copy=False).copy()
            else:
                self._running_sum_descriptor += rec.descriptor.astype(np.float32, copy=False)
        elif self.descriptor_agg == "ema":
            if self._ema_descriptor is None:
                self._ema_descriptor = rec.descriptor.astype(np.float32, copy=False).copy()
            else:
                self._ema_descriptor = self.ema_decay * self._ema_descriptor + (
                    1.0 - self.ema_decay
                ) * rec.descriptor.astype(np.float32, copy=False)

        self._records.append(rec)
        if len(self._records) > self.max_window:
            popped = self._records.popleft()
            if self.descriptor_agg == "mean":
                # subtract from running sum
                self._running_sum_descriptor -= popped.descriptor.astype(np.float32, copy=False)
            # EMA does not support exact removal; keep as-is (acceptable approximation)

    def _aggregate_descriptor(self) -> np.ndarray:
        if not self._records:
            return np.empty((0,), dtype=np.float32)
        if self.descriptor_agg == "last":
            return self._records[-1].descriptor.astype(np.float32, copy=False)
        if self.descriptor_agg == "ema":
            # If only one frame, ema == that descriptor
            if self._ema_descriptor is not None:
                return self._ema_descriptor.astype(np.float32, copy=False)
            return self._records[-1].descriptor.astype(np.float32, copy=False)
        # mean
        count = float(len(self._records))
        if self._running_sum_descriptor is None:
            return self._records[-1].descriptor.astype(np.float32, copy=False)
        return (self._running_sum_descriptor / count).astype(np.float32, copy=False)


# =============================================================================
# Registration Pipeline
# =============================================================================


class RansacPointCloudRegistrationPipeline:
    """Point cloud registration pipeline using Open3D RANSAC.

    The pipeline performs the following steps:
      1) Voxel downsample both clouds and estimate normals
      2) Compute FPFH features
      3) Run RANSAC-based global registration

    Returned transform semantics:
    - We feed Open3D with `source=query`, `target=database`.
    - Open3D returns a transform mapping `source→target`, i.e. `T_db<-q`.
    This can be composed with a known database world pose `T_w<-db` to get
    `T_w<-q = T_w<-db * T_db<-q`.
    """

    def __init__(self, voxel_downsample_size: float = 0.5) -> None:
        """Initialize the RANSAC registration pipeline.

        Args:
            voxel_downsample_size: Voxel size used for downsampling and
                for computing normal/feature radii. Larger values are faster but
                may reduce accuracy. Defaults to 0.5.
        """
        self.voxel_downsample_size = voxel_downsample_size

    def _preprocess_point_cloud(
        self, points: Tensor
    ) -> tuple[o3d.geometry.PointCloud, o3d.pipelines.registration.Feature]:
        """Downsample point cloud and compute FPFH features.

        Args:
            points: Tensor [N,3] float32.

        Returns:
            Tuple of (downsampled Open3D point cloud, FPFH feature).
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points.cpu().numpy())
        pcd_down = pcd.voxel_down_sample(self.voxel_downsample_size)
        radius_normal = self.voxel_downsample_size * 2
        pcd_down.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))
        radius_feature = self.voxel_downsample_size * 5
        pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd_down, o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100)
        )
        return pcd_down, pcd_fpfh

    def _execute_global_registration(
        self,
        source_down: o3d.geometry.PointCloud,
        target_down: o3d.geometry.PointCloud,
        source_fpfh: o3d.pipelines.registration.Feature,
        target_fpfh: o3d.pipelines.registration.Feature,
    ) -> o3d.pipelines.registration.RegistrationResult:
        """Run Open3D RANSAC-based registration."""
        distance_threshold = self.voxel_downsample_size * 1.5
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source_down,
            target_down,
            source_fpfh,
            target_fpfh,
            True,
            distance_threshold,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            3,
            [
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold),
            ],
            o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999),
        )
        return result

    def infer(self, query_pc: Tensor, db_pc: Tensor) -> RegistrationResult:
        """Estimate rigid transform that maps query into the database frame.

        Args:
            query_pc: Tensor [N,3] float32 for the query cloud.
            db_pc: Tensor [M,3] float32 for the database cloud.

        Returns:
            RegistrationResult: Result structure with transformation and metrics.

        Notes:
            To obtain the world pose of the query from a known `T_w<-db`, use
            `T_w<-q = T_w<-db * T_db<-q`.
        """
        source_down, source_fpfh = self._preprocess_point_cloud(query_pc)
        target_down, target_fpfh = self._preprocess_point_cloud(db_pc)
        result = self._execute_global_registration(source_down, target_down, source_fpfh, target_fpfh)
        # Extract metrics when available
        fitness: float | None
        inlier_rmse: float | None
        try:
            fitness = float(result.fitness)
        except Exception:
            fitness = None
        try:
            inlier_rmse = float(result.inlier_rmse)
        except Exception:
            inlier_rmse = None

        num_inliers: int | None = None
        if hasattr(result, "correspondence_set"):
            try:
                num_inliers = int(len(result.correspondence_set))
            except Exception:
                num_inliers = None

        success = bool(fitness is not None and fitness > 0.0)

        return RegistrationResult(
            transformation=result.transformation,
            success=success,
            fitness=fitness,
            inlier_rmse=inlier_rmse,
            num_inliers=num_inliers,
        )


# =============================================================================
# Localization Pipeline
# =============================================================================


class LocalizationPipeline:
    """Top-k localization using PR candidates and point cloud registration.

    Overview:
    - Run top-k Place Recognition to retrieve database candidates
    - Load each candidate's database point cloud via the index `pointcloud_path`
    - Run registration to estimate a rigid transform between the query and
      database clouds
    - Compose a world pose for the query for each candidate and select the best

    Transform direction and composition (explicit):
    - Registration must return a transform that maps query→database, i.e. `T_db<-q`
      (Open3D returns source→target; we pass source=query, target=database).
    - If the database world pose is `T_w<-db`, the query world pose is computed as
      `T_w<-q = T_w<-db · T_db<-q` (homogeneous 4×4 with column vectors).
    """

    def __init__(
        self,
        index: Index,
        place_recognition: PlaceRecognitionPipeline,
        registration: object,
        index_root: str | Path,
        require_db_pointcloud: bool = False,
    ) -> None:
        """Initialize the localization pipeline.

        Args:
            index: Retrieval index that provides metadata mapping
                `(db_idx, db_pose, db_pointcloud_path)` for row positions.
            place_recognition: Pipeline to obtain top-k candidates and raw
                distances for a query.
            registration: Registration component exposing
                `infer(query_pc: Tensor, db_pc: Tensor) -> RegistrationResult`.
            index_root: Root directory for resolving relative database point
                cloud paths from the index metadata.
            require_db_pointcloud: If True, missing/invalid DB point cloud paths
                cause a FileNotFoundError; if False, such candidates are skipped.
                Defaults to False.
        """
        self.index = index
        self.pr = place_recognition
        self.reg = registration
        self.store = PointCloudStore(root_dir=Path(index_root))
        self.require_db_pointcloud = require_db_pointcloud

    @torch.inference_mode()
    def infer(
        self,
        pr_input: dict[str, Tensor],
        query_pc: Tensor,
        k: int = 5,
    ) -> LocalizationResult:
        """Run localization for a single query.

        Args:
            pr_input: Dict of tensors for the PR model input.
            query_pc: Raw query point cloud tensor [N,3] for registration.
            k: Number of PR candidates to evaluate.

        Returns:
            LocalizationResult: Per-candidate estimated poses and the chosen match.

        Raises:
            FileNotFoundError: When `require_db_pointcloud=True` and a candidate
                has a missing/invalid `pointcloud_path` or the file is empty.
            RuntimeError: When no valid candidates remain after loading DB point clouds.
        """
        # Run PR
        pr_res: PlaceRecognitionResult = self.pr.infer(pr_input, k=k)
        inds = pr_res.indices
        dists = pr_res.distances
        db_idx, db_pose, db_pc_path = self.index.get_meta(inds)

        candidates: list[LocalizedCandidate] = []

        for i in range(inds.shape[0]):
            # Load DB PC if available
            rel_path_obj = db_pc_path[i]
            rel_path: str | None
            if isinstance(rel_path_obj, str):
                rel_path = rel_path_obj
            else:
                rel_path = None

            if rel_path is None:
                if self.require_db_pointcloud:
                    raise FileNotFoundError("Database pointcloud path is missing for a candidate")
                # Skip candidate
                continue

            db_pc = self.store.load(rel_path)
            if db_pc.numel() == 0:
                if self.require_db_pointcloud:
                    raise FileNotFoundError("Database pointcloud file is missing or empty")
                continue

            # Run registration: Open3D-style source→target, i.e. T_db<-q (query→database)
            reg_res = self.reg.infer(query_pc=query_pc, db_pc=db_pc)
            T_db_from_q = reg_res.transformation if hasattr(reg_res, "transformation") else reg_res
            # Compose world pose of query: T_w<-q = T_w<-db · T_db<-q
            T_db = _pose7_to_matrix(db_pose[i])
            T_est = T_db @ T_db_from_q
            est_pose = _matrix_to_pose7(T_est)

            candidate = LocalizedCandidate(
                idx=int(db_idx[i]),
                pr_distance=float(dists[i]),
                db_pose=db_pose[i],
                db_pointcloud_path=rel_path,
                estimated_pose=est_pose,
                registration_confidence=1.0,
            )
            candidates.append(candidate)

        if not candidates:
            raise RuntimeError("No valid candidates after loading database point clouds")

        # Choose best by registration_confidence (all equal -> fallback to smallest PR distance)
        best = min(candidates, key=lambda c: (-c.registration_confidence, c.pr_distance))
        return LocalizationResult(version="1", candidates=candidates, chosen_idx=best.idx)

