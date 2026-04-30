"""FAISS-based retrieval index for place recognition.

This module provides a minimal, high-performance FAISS Flat index built at load
time from `descriptors.npy`, alongside utilities to read associated metadata.

On-disk layout (required):
- `descriptors.npy` (float32 [N,D])
- `meta.parquet` (required columns: `idx:int`, `pose:[7]`; optional:
  `pointcloud_path:str|NaN` with values like `scans/000227.pcd` or
  `scans/000227.bin` relative to the index root)
- `schema.json` (versioned)

This code is based on the OpenPlaceRecognition library (Apache 2.0 License).
Source: https://github.com/OPR-Project/OpenPlaceRecognition
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Dict, Optional
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from loguru import logger
import opr
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import faiss
except Exception as e:
    raise ImportError(
        "FAISS is required for FaissFlatIndex. Please install faiss-cpu or faiss-gpu."
    ) from e


def _infer_model_device(model: nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def _move_collated_batch_to_device(batch: Dict[str, Any], device: torch.device) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    non_blocking = device.type == "cuda"
    for k, v in batch.items():
        if v is None:
            out[k] = None
        elif hasattr(v, "to"):
            out[k] = v.to(device, non_blocking=non_blocking)
        else:
            out[k] = v
    return out


def _descriptor_tensor_from_output(out: Any) -> torch.Tensor:
    if isinstance(out, dict):
        if "final_descriptor" not in out:
            raise KeyError("Model output must contain key 'final_descriptor'")
        fd = out["final_descriptor"]
        if isinstance(fd, dict) and "final_descriptor" in fd:
            fd = fd["final_descriptor"]
        if not isinstance(fd, torch.Tensor):
            raise TypeError(f"final_descriptor must be a torch.Tensor, got {type(fd)}")
        return fd
    if isinstance(out, torch.Tensor):
        return out
    raise TypeError(f"Unexpected model output type: {type(out)}")


# =============================================================================
# Enums and Schema
# =============================================================================


class IndexMetric(str, Enum):
    """Distance/Similarity metric used by the index."""

    L2 = "l2"
    IP = "ip"  # inner product


@dataclass(frozen=True)
class IndexSchema:
    """Schema information loaded from schema.json."""

    version: str
    dim: int
    metric: IndexMetric
    created_at: str
    opr_version: str | None
    descriptors_sha256: str | None


# =============================================================================
# Abstract Base Class
# =============================================================================


class Index(ABC):
    """Abstract base for retrieval index backends."""

    @classmethod
    @abstractmethod
    def load(cls, directory: str | Path) -> "Index":
        """Load index from a directory containing descriptors, meta and schema."""

    @abstractmethod
    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search top-k.

        Args:
            queries: float32 array of shape [Q, D].
            k: number of nearest neighbors to return.

        Returns:
            indices: int64 array [Q, k] of internal row positions (0..N-1).
            distances: float32 array [Q, k] of raw backend distances.
        """

    @abstractmethod
    def size(self) -> int:
        """Number of database items (N)."""

    @abstractmethod
    def dim(self) -> int:
        """Descriptor dimensionality (D)."""

    @abstractmethod
    def metric(self) -> IndexMetric:
        """Metric used (L2 or IP)."""

    @abstractmethod
    def get_meta(self, row_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Fetch aligned metadata for given row positions.

        Args:
            row_positions: Array of internal row indices (0..N-1) to fetch metadata for.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]: Tuple of
                `(db_idx, db_pose, db_pointcloud_path)` where:
                - `db_idx` is int64 array of shape [M] with dataset item ids.
                - `db_pose` is float32 array of shape [M, 7] with poses
                  in order `tx, ty, tz, qx, qy, qz, qw`.
                - `db_pointcloud_path` is object array of shape [M] with each
                  element being a relative path string like "scans/000227.pcd"
                  or "scans/000227.bin", or `numpy.nan` when not available.
        """

    @abstractmethod
    def distances_to_rows(self, queries: np.ndarray, row_positions: np.ndarray) -> np.ndarray:
        """Raw distances from each query to database rows (same convention as ``search``).

        Args:
            queries: float32 [Q, D].
            row_positions: int64 [Q, K] internal row ids (0..N-1).

        Returns:
            float32 [Q, K] distances aligned with ``row_positions``.
        """


# =============================================================================
# IO Helpers
# =============================================================================


def _sha256_file(path: Path, chunk_size: int = 2**20) -> str:
    """Compute SHA256 hash of a file."""
    sha = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha.update(chunk)
    return sha.hexdigest()


def load_descriptors(path: str | Path, mmap: bool = True) -> np.ndarray:
    """Load descriptors.npy as float32 [N, D].

    Args:
        path: Path to `descriptors.npy`.
        mmap: If True, memory-map the array in read-only mode.

    Returns:
        np.ndarray: Float32 array of shape [N, D].

    Raises:
        ValueError: If the loaded array is not 2D.
    """
    arr = np.load(str(path), mmap_mode="r" if mmap else None)
    if arr.dtype != np.float32:
        arr = arr.astype(np.float32, copy=False)
    if arr.ndim != 2:
        raise ValueError(f"descriptors at {path} must be 2D; got shape {arr.shape}")
    return arr


def load_meta(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
    """Load meta.parquet and return (db_idx [N], db_pose [N,7], pointcloud_path [N], full_df).

    Args:
        path: Path to `meta.parquet`.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, pd.DataFrame]:
            `(db_idx, db_pose, db_pointcloud_path, df)`.

    Raises:
        ValueError: If required columns are missing or pose length is not 7.
    """
    df = pd.read_parquet(path)
    if "idx" not in df.columns:
        raise ValueError("meta.parquet must contain 'idx' column")
    if "pose" not in df.columns:
        raise ValueError("meta.parquet must contain 'pose' column (length 7)")
    poses = df["pose"].to_numpy()
    # Ensure each pose element is length-7
    pose_arr = np.stack([np.asarray(p, dtype=np.float32) for p in poses], axis=0)
    if pose_arr.shape[1] != 7:
        raise ValueError(f"pose must be length 7; got shape {pose_arr.shape}")
    idx_arr = df["idx"].to_numpy(dtype=np.int64)

    # Optional pointcloud_path column: relative path to pointcloud file (pcd/bin) or NaN
    if "pointcloud_path" in df.columns:
        paths_series = df["pointcloud_path"]

        # Normalize: allow NaN/None; otherwise expect str ending with .pcd or .bin
        def _validate_path(val: object) -> object:
            if pd.isna(val):
                return np.nan
            if isinstance(val, str):
                lower = val.lower()
                if (lower.endswith(".pcd") or lower.endswith(".bin")) and not Path(val).is_absolute():
                    return val
            raise ValueError(
                "meta.parquet 'pointcloud_path' must be a relative str ending with .pcd or .bin, or NaN"
            )

        db_pc_path = np.array([_validate_path(v) for v in paths_series.to_list()], dtype=object)
    else:
        # If missing, fill with NaN object array for compatibility
        db_pc_path = np.array([f"scans/{j:06d}.pcd" for j in df["idx"]], dtype=object)#       [np.nan] * idx_arr.shape[0], dtype=object)

    return idx_arr, pose_arr, db_pc_path, df


def load_schema(path: str | Path) -> IndexSchema:
    """Load schema.json into IndexSchema.

    Args:
        path: Path to `schema.json`.

    Returns:
        IndexSchema: Parsed schema information.
    """
    with Path(path).open("r") as f:
        data = json.load(f)
    metric_raw = data.get("metric")
    metric = IndexMetric(metric_raw) if metric_raw is not None else IndexMetric.L2
    return IndexSchema(
        version=str(data.get("version", "1")),
        dim=int(data["dim"]),
        metric=metric,
        created_at=str(data.get("created_at", "")),
        opr_version=str(data.get("opr_version", "")) if data.get("opr_version") is not None else None,
        descriptors_sha256=(
            str(data.get("descriptors_sha256", "")) if data.get("descriptors_sha256") is not None else None
        ),
    )


def validate_files(base_dir: str | Path) -> tuple[Path, Path, Path]:
    """Validate required files exist and return their paths.

    Args:
        base_dir: Directory containing index files.

    Returns:
        tuple[Path, Path, Path]: Paths to `(descriptors.npy, meta.parquet, schema.json)`.

    Raises:
        FileNotFoundError: If any of the required files is missing.
    """
    base = Path(base_dir)
    desc = base / "descriptors.npy"
    meta = base / "meta.parquet"
    schema = base / "schema.json"
    for p in (desc, meta, schema):
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")
    return desc, meta, schema


def validate_consistency(
    descriptors: np.ndarray, db_idx: np.ndarray, schema: IndexSchema, desc_path: Path
) -> None:
    """Validate shapes and optional hash consistency between files.

    Args:
        descriptors: Float32 [N, D] descriptors array.
        db_idx: Int64 [N] dataset ids from meta.
        schema: Parsed schema.
        desc_path: Path to descriptors file for hash calculation when provided.

    Raises:
        ValueError: If shapes/lengths mismatch or descriptors hash does not match schema.
    """
    if descriptors.shape[0] != db_idx.shape[0]:
        raise ValueError(
            f"Row count mismatch: descriptors N={descriptors.shape[0]} vs meta N={db_idx.shape[0]}"
        )
    if descriptors.shape[1] != schema.dim:
        raise ValueError(f"Dim mismatch: descriptors D={descriptors.shape[1]} vs schema dim={schema.dim}")
    # Optional hash validation
    if schema.descriptors_sha256:
        file_hash = _sha256_file(desc_path)
        if file_hash != schema.descriptors_sha256:
            raise ValueError("descriptors hash mismatch: schema.descriptors_sha256 does not match file")


# =============================================================================
# FAISS Flat Index Implementation
# =============================================================================


class FaissFlatIndex(Index):
    """FAISS Flat backend using L2 or IP metric.

    Builds an in-memory `faiss.IndexFlatL2` or `faiss.IndexFlatIP` at load time
    from `descriptors.npy` and exposes a minimal search API returning raw
    distances and row positions.
    """

    def __init__(
        self,
        descriptors: np.ndarray,
        db_idx: np.ndarray,
        db_pose: np.ndarray,
        db_pointcloud_path: np.ndarray,
        schema: IndexSchema,
    ) -> None:
        """Initialize the index instance.

        Args:
            descriptors: Float32 array [N, D] of database descriptors.
            db_idx: Int64 array [N] of dataset item ids aligned with descriptors.
            db_pose: Float32 array [N, 7] of poses aligned with descriptors.
            db_pointcloud_path: Object array [N] of relative point cloud paths
                (e.g., "scans/000227.pcd" or "scans/000227.bin") or NaN values
                when absent.
            schema: Parsed `IndexSchema` with dim/metric.
        """
        self._schema = schema
        self._descriptors = np.ascontiguousarray(descriptors.astype(np.float32, copy=False))
        self._db_idx = db_idx.astype(np.int64, copy=False)
        self._db_pose = db_pose.astype(np.float32, copy=False)
        # keep as object array to preserve NaN or str values
        self._db_pointcloud_path = db_pointcloud_path.astype(object, copy=False)

        d = self._descriptors.shape[1]
        if self._schema.metric == IndexMetric.IP:
            self._index = faiss.IndexFlatIP(d)
        else:
            self._index = faiss.IndexFlatL2(d)
        self._index.add(self._descriptors)

    @classmethod
    def load(cls, directory: str | Path) -> "FaissFlatIndex":
        """Load index from a directory with descriptors/meta/schema.

        Args:
            directory: Path containing `descriptors.npy`, `meta.parquet`, `schema.json`.

        Returns:
            FaissFlatIndex: Loaded index ready for search.
        """
        desc_path, meta_path, schema_path = validate_files(directory)
        descriptors = load_descriptors(desc_path)
        db_idx, db_pose, db_pc_path, _ = load_meta(meta_path)
        schema = load_schema(schema_path)
        validate_consistency(descriptors, db_idx, schema, Path(desc_path))
        return cls(
            descriptors=descriptors,
            db_idx=db_idx,
            db_pose=db_pose,
            db_pointcloud_path=db_pc_path,
            schema=schema,
        )

    @classmethod
    def generate(
        cls,
        directory: str | Path,
        dataset: Optional[Dataset] = None,
        dataloader: Optional[DataLoader] = None,
        model: Optional[nn.Module] = None,
        rebuild_meta: bool = False,
        rebuild_descriptors: bool = False,
        batch_size: int = 16,
        num_workers: int = 4,
        shuffle: bool = False,
        metric: str = "l2", # can be also "ip" - inner product
        version: Any = 1
    ) -> "FaissFlatIndex":
        """Generate index files (descriptors/meta/schema) in directory based on Dataset and model.

        Args:
            directory: Path where files `descriptors.npy`, `meta.parquet`, `schema.json` are need to be created.
            dataset: object of class that is based on torch Dataset and contains "save_meta_parquet" and "collate_fn" functions that parse dataset.
            dataloader: 

        Returns:
            FaissFlatIndex: Loaded index ready for search.
        """
        Path(directory).mkdir(parents=True, exist_ok=True)

        meta_exists = (Path(directory) / "meta.parquet").exists()
        descriptors_exists = (Path(directory) / "descriptors.npy").exists()
        rebuild_meta_needed = not meta_exists or rebuild_meta
        rebuild_descriptors_needed = not descriptors_exists or rebuild_descriptors
        
        if dataset is None and rebuild_meta_needed:
            logger.info("Can't build meta.parquet file. Have no dataset object")
        elif rebuild_meta_needed:
            dataset.save_meta_parquet(directory, "meta.parquet")
            logger.info(f"meta.parquet file was saved in {directory}")
        else:
            logger.info("Using existing meta.parquet")
        
        if rebuild_descriptors_needed:
            if model is None:
                logger.info("Can't build descriptors.npy file. Have no model")
            elif dataloader is None and dataset is None:
                logger.info("Can't build descriptors.npy file. Have no dataset or dataloader object")
            else:
                if dataloader is None:
                    dataloader = DataLoader(
                        dataset, 
                        batch_size=batch_size, 
                        shuffle=shuffle, 
                        num_workers=num_workers, 
                        collate_fn=dataset.collate_fn
                    )
                descriptors = []
                device = _infer_model_device(model)
                with torch.no_grad():
                    for batch in tqdm(dataloader):
                        batch = _move_collated_batch_to_device(batch, device)
                        raw = model(batch)
                        final_descriptor = _descriptor_tensor_from_output(raw)
                        descriptors.append(final_descriptor.detach().cpu().numpy())
                descriptors = np.concatenate(descriptors, axis=0)
                
                np.save(f"{directory}/descriptors.npy", descriptors)
                logger.info(f"descriptors.npy file was saved in {directory}")
        else:
            logger.info("Using existing descriptors.npy")

        meta_exists = (Path(directory) / "meta.parquet").exists()
        descriptors_exists = (Path(directory) / "descriptors.npy").exists()
        if not (descriptors_exists and meta_exists):
            logger.info("Can't build schema.json file. Have no descriptors or meta files")
        else:
            descriptors = np.load(Path(directory) / "descriptors.npy")
            N, D = descriptors.shape
            schema = {
                "version": str(version),
                "number": N,
                "dim": D, 
                "metric": metric, 
                "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), 
                "opr_version": opr.__version__}

            Path(f"{directory}/schema.json").write_text(json.dumps(schema))
            logger.info(f"schema.json file was saved in {directory}")

        return cls.load(directory)

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        """Search top-k nearest neighbors.

        Args:
            queries: Float32 array [Q, D] of query descriptors.
            k: Number of neighbors to return.

        Returns:
            tuple[np.ndarray, np.ndarray]: `(indices, distances)` where `indices`
            is int64 array [Q, k] of row positions and `distances` is float32
            array [Q, k] of raw FAISS distances.
        """
        q = np.ascontiguousarray(queries.astype(np.float32, copy=False))
        distances, inds = self._index.search(q, k)
        return inds.astype(np.int64, copy=False), distances.astype(np.float32, copy=False)

    def distances_to_rows(self, queries: np.ndarray, row_positions: np.ndarray) -> np.ndarray:
        """Pairwise distances matching FAISS Flat (L2 squared, or negated IP)."""
        q = np.ascontiguousarray(queries.astype(np.float32, copy=False))
        rows = row_positions.astype(np.int64, copy=False)
        if q.ndim != 2 or rows.ndim != 2:
            raise ValueError(f"Expected queries [Q,D] and row_positions [Q,K]; got {q.shape}, {rows.shape}")
        if q.shape[0] != rows.shape[0]:
            raise ValueError(
                f"Batch mismatch: queries Q={q.shape[0]} vs row_positions Q={rows.shape[0]}"
            )
        db_vecs = self._descriptors[rows]
        if self._schema.metric == IndexMetric.L2:
            diff = q[:, np.newaxis, :] - db_vecs
            return np.sum(diff * diff, axis=-1, dtype=np.float32)
        ip = np.sum(q[:, np.newaxis, :] * db_vecs, axis=-1, dtype=np.float32)
        return (-ip).astype(np.float32, copy=False)

    def size(self) -> int:
        """Return number of database items (N)."""
        return self._descriptors.shape[0]

    def dim(self) -> int:
        """Return descriptor dimensionality (D)."""
        return self._descriptors.shape[1]

    def metric(self) -> IndexMetric:
        """Return metric used by the index (L2 or IP)."""
        return self._schema.metric

    def get_meta(self, row_positions: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Map row positions to dataset indices and poses.

        Args:
            row_positions: Int64 array [M] of internal row positions (0..N-1).

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                `(db_idx, db_pose, db_pointcloud_path)` aligned with input rows.
        """
        rows = row_positions.astype(np.int64, copy=False)
        return self._db_idx[rows], self._db_pose[rows], self._db_pointcloud_path[rows]

