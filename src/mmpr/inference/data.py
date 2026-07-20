"""Result dataclasses for inference pipelines.

This module contains all result data structures returned by the inference
pipelines including place recognition, registration, and localization results.

This code is based on the OpenPlaceRecognition library (Apache 2.0 License).
Source: https://github.com/OPR-Project/OpenPlaceRecognition
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class PlaceRecognitionResult:
    """Result of a top-k place recognition query.

    Notes:
        - ``indices`` are internal row ids in the index (shape [k]).
        - ``db_idx`` and ``db_pose`` are optional to allow caching per-frame
          results without performing metadata lookups for each frame. Sequence
          pipelines may fill only for the fused final result.
    """

    descriptor: np.ndarray  # [D]
    indices: np.ndarray  # [k] internal row positions
    distances: np.ndarray  # [k] raw distances
    db_idx: np.ndarray | None = None  # [k] dataset ids
    db_pose: np.ndarray | None = None  # [k,7] poses (tx,ty,tz,qx,qy,qz,qw)
    rerank_descriptor: np.ndarray | None = None  # [D] rerank descriptor

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "descriptor": self.descriptor.tolist(),
            "rerank_descriptor": self.rerank_descriptor.tolist() if self.rerank_descriptor is not None else None,
            "indices": self.indices.tolist(),
            "distances": self.distances.tolist(),
            "db_idx": self.db_idx.tolist() if self.db_idx is not None else None,
            "db_pose": self.db_pose.tolist() if self.db_pose is not None else None,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "PlaceRecognitionResult":
        """Create instance from a dictionary."""
        return cls(
            descriptor=np.array(data["descriptor"], dtype=np.float32),
            rerank_descriptor=np.array(data["rerank_descriptor"], dtype=np.float32) if data.get("rerank_descriptor") is not None else None,
            indices=np.array(data["indices"], dtype=np.int64),
            distances=np.array(data["distances"], dtype=np.float32),
            db_idx=np.array(data["db_idx"], dtype=np.int64) if data.get("db_idx") is not None else None,
            db_pose=np.array(data["db_pose"], dtype=np.float32) if data.get("db_pose") is not None else None,
        )

    def save(self, path: str | Path) -> None:
        """Save result to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "PlaceRecognitionResult":
        """Load result from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


@dataclass
class SequencePRDebug:
    """Optional debug information returned by the sequence pipeline."""

    per_frame_indices: np.ndarray  # [N, per_k]
    per_frame_distances: np.ndarray  # [N, per_k]
    fused_indices: np.ndarray  # [final_k]
    fused_distances: np.ndarray  # [final_k]
    window_size: int
    descriptor_agg: str


@dataclass
class RegistrationResult:
    """Result of point cloud registration mapping query→database.

    Attributes:
        transformation: 4×4 matrix `T_db<-q` such that `x_db = T_db<-q * x_q`.
        success: Whether registration likely succeeded (heuristic).
        fitness: Overlap ratio reported by Open3D (if available).
        inlier_rmse: Inlier root mean square error (if available).
        num_inliers: Number of inlier correspondences (if available).
    """

    transformation: np.ndarray
    success: bool
    fitness: float | None = None
    inlier_rmse: float | None = None
    num_inliers: int | None = None

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "transformation": self.transformation.tolist(),
            "success": self.success,
            "fitness": self.fitness,
            "inlier_rmse": self.inlier_rmse,
            "num_inliers": self.num_inliers,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "RegistrationResult":
        """Create instance from a dictionary."""
        return cls(
            transformation=np.array(data["transformation"], dtype=np.float64),
            success=bool(data["success"]),
            fitness=float(data["fitness"]) if data.get("fitness") is not None else None,
            inlier_rmse=float(data["inlier_rmse"]) if data.get("inlier_rmse") is not None else None,
            num_inliers=int(data["num_inliers"]) if data.get("num_inliers") is not None else None,
        )

    def save(self, path: str | Path) -> None:
        """Save result to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "RegistrationResult":
        """Load result from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)


@dataclass
class LocalizedCandidate:
    """Per-candidate localization result."""

    idx: int
    pr_distance: float
    db_pose: np.ndarray  # [7]
    db_pointcloud_path: str | None
    estimated_pose: np.ndarray  # [7]
    registration_confidence: float

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "idx": self.idx,
            "pr_distance": self.pr_distance,
            "db_pose": self.db_pose.tolist(),
            "db_pointcloud_path": self.db_pointcloud_path,
            "estimated_pose": self.estimated_pose.tolist(),
            "registration_confidence": self.registration_confidence,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LocalizedCandidate":
        """Create instance from a dictionary."""
        return cls(
            idx=int(data["idx"]),
            pr_distance=float(data["pr_distance"]),
            db_pose=np.array(data["db_pose"], dtype=np.float32),
            db_pointcloud_path=data.get("db_pointcloud_path"),
            estimated_pose=np.array(data["estimated_pose"], dtype=np.float64),
            registration_confidence=float(data["registration_confidence"]),
        )


@dataclass
class LocalizationResult:
    """Full localization result."""

    version: str
    candidates: list[LocalizedCandidate]
    chosen_idx: int

    def to_dict(self) -> dict:
        """Convert to a JSON-serializable dictionary."""
        return {
            "version": self.version,
            "candidates": [c.to_dict() for c in self.candidates],
            "chosen_idx": self.chosen_idx,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "LocalizationResult":
        """Create instance from a dictionary."""
        return cls(
            version=str(data["version"]),
            candidates=[LocalizedCandidate.from_dict(c) for c in data["candidates"]],
            chosen_idx=int(data["chosen_idx"]),
        )

    def save(self, path: str | Path) -> None:
        """Save result to a JSON file."""
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "LocalizationResult":
        """Load result from a JSON file."""
        data = json.loads(Path(path).read_text())
        return cls.from_dict(data)
