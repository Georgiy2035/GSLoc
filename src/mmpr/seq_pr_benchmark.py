from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from opr.inference.index import FaissFlatIndex

from mmpr.pr_cache import load_pr_cache_npz
from mmpr.sequence_emulator import EmulatorConfig, emulate_sequence_fusion
from mmpr.metrics import recall_at_k_per_query, build_micro_curves_from_rankings
from mmpr.data.transforms import get_T_map_to_world


@dataclass
class SequenceBenchmarkConfig:
    q_df: pd.Dataframe
    db_index_dir: Path
    cache_path: Path
    root_data_dir: Path
    map_name: str
    max_window: int = 20
    per_frame_k_used: int | None = None
    final_k: int = 25
    recency_weighting: Literal["none", "linear", "exp"] = "none"
    recall_threshold_m: float = 5.0


@dataclass
class BenchmarkArtifacts:
    recall_at_k: dict[int, float]
    auc_pr: float
    f1_max: float
    max_recall_at_prec1: float
    num_queries_total: int
    num_queries_valid: int
    curves: dict[str, np.ndarray]
    fused_rankings: tuple[np.ndarray, np.ndarray] | None


class SequencePRBenchmarker:
    def __init__(self, cfg: SequenceBenchmarkConfig) -> None:
        self.cfg = cfg
        self.index: FaissFlatIndex | None = None
        self.frames = None
        self.query_xyz: np.ndarray | None = None
        self.valid_indices: list[int] = []

    # --- data loading helpers ---
    def _load_query_xyz(self) -> np.ndarray:
        map_dir = (self.cfg.root_data_dir / self.cfg.map_name / "keyframe_map").resolve()
        traj_path = map_dir / "poses.csv"
        vals = np.genfromtxt(str(traj_path), delimiter=",", comments="#", dtype=np.float32)
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        xyz = vals[:, 1:4].astype(np.float32, copy=False)
        quat = vals[:, 4:8].astype(np.float32, copy=False)
        T_m2w = get_T_map_to_world(self.cfg.map_name)
        out = np.zeros_like(xyz, dtype=np.float32)
        for i in range(xyz.shape[0]):
            Rm = Rotation.from_quat(quat[i]).as_matrix().astype(np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rm
            T[:3, 3] = xyz[i]
            Tw = T_m2w @ T
            out[i] = Tw[:3, 3]
        return out

    def _build_valid_subset(self, query_xyz: np.ndarray, index: FaissFlatIndex) -> list[int]:
        """Build valid subset of queries.
        
        We consider a query valid if it has at least one database pose within the recall threshold.

        Args:
            query_xyz: (N, 3) array of query positions
            index: FaissFlatIndex

        Returns:
            list[int]: valid query indices
        """
        thr = float(self.cfg.recall_threshold_m)
        valid: list[int] = []
        db_all_xyz = None
        try:
            db_all_xyz = index._db_pose[:, :3]  # type: ignore[attr-defined]
        except Exception:
            db_all_xyz = None
        if isinstance(db_all_xyz, np.ndarray):
            for i, q in enumerate(query_xyz):
                d = np.linalg.norm(db_all_xyz - q[None, :], axis=1)
                if np.any(d < thr):
                    valid.append(i)
        else:
            valid = list(range(query_xyz.shape[0]))
        return valid

    # --- main API ---
    def run(self) -> BenchmarkArtifacts:
        """Benchmark Sequence Place Recognition (micro-averaged metrics)"""
        # Load cache and index
        self.frames = load_pr_cache_npz(self.cfg.cache_path)
        self.index = FaissFlatIndex.load(str(self.cfg.db_index_dir))

        # Build GT query positions and valid subset
        self.query_xyz = np.array([[self.cfg.q_df["scene_id"][i], 0, 0] for i in range(len(self.cfg.q_df))], dtype=np.float32)#self._load_query_xyz()
        if len(self.query_xyz) > len(self.frames):
            self.query_xyz = self.query_xyz[:len(self.frames)]
        if len(self.query_xyz) != len(self.frames):
            raise RuntimeError(f"#query_xyz ({len(self.query_xyz)}) != #frames ({len(self.frames)}) in {self.cfg.cache_path}")
        self.valid_indices = self._build_valid_subset(self.query_xyz, self.index)

        # Emulate sequence fusion
        cfg = EmulatorConfig(
            max_window=int(self.cfg.max_window),
            per_frame_k_used=(None if self.cfg.per_frame_k_used is None else int(self.cfg.per_frame_k_used)),
            final_k=int(self.cfg.final_k),
            recency_weighting=str(self.cfg.recency_weighting),
        )
        fused = emulate_sequence_fusion(self.frames, cfg)

        # Build rankings and db_idx->xyz cache
        db_xyz_by_db_idx: dict[int, np.ndarray] = {}
        rankings: list[tuple[np.ndarray, np.ndarray]] = []
        for fused_d, fused_i in fused:
            if fused_i.size == 0:
                rankings.append((np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)))
                continue
            db_idx, db_pose, _db_pc = self.index.get_meta(fused_i)
            for j in range(db_idx.shape[0]):
                did = int(db_idx[j])
                if did not in db_xyz_by_db_idx:
                    db_xyz_by_db_idx[did] = np.asarray(db_pose[j][:3], dtype=np.float32)
            rankings.append((fused_d, db_idx))

        thr = float(self.cfg.recall_threshold_m)

        def oracle(query_id: int, db_idx: int) -> bool:
            if not (0 <= query_id < self.query_xyz.shape[0]):
                return False
            q = self.query_xyz[int(query_id)]
            d = db_xyz_by_db_idx.get(int(db_idx))
            if d is None:
                return False
            return float(np.linalg.norm(q - d)) < thr

        # Recall@K (K in [1..25]) on valid subset
        Ks = list(range(1, 26))
        recalls = {k: [] for k in Ks}
        for qid in self.valid_indices:
            dists, db_ids = rankings[qid]
            is_pos = np.array([oracle(qid, int(db)) for db in db_ids], dtype=bool)
            stats = recall_at_k_per_query(db_ids, is_pos, Ks)
            for k in Ks:
                recalls[k].append(stats[k])
        recall_at_k = {k: float(np.mean(recalls[k]) if len(recalls[k]) > 0 else 0.0) for k in Ks}

        # Micro PR curves/metrics on valid subset
        filtered_rankings = [rankings[i] for i in self.valid_indices]
        filtered_query_ids = np.asarray(self.valid_indices, dtype=np.int64)
        curves = build_micro_curves_from_rankings(filtered_rankings, oracle, query_ids=filtered_query_ids)

        # Prepare artifacts (curves and padded fused rankings)
        curves_dict = {
            "precision": curves.precision,
            "recall": curves.recall,
            "thresholds": curves.thresholds,
            "f1": curves.f1,
        }
        max_len = max((len(r[0]) for r in rankings), default=0)
        fused_np: tuple[np.ndarray, np.ndarray] | None
        if max_len > 0:
            D = np.full((len(rankings), max_len), np.inf, dtype=np.float32)
            I = np.full((len(rankings), max_len), -1, dtype=np.int64)
            for i, (d, db) in enumerate(rankings):
                L = min(len(d), max_len)
                D[i, :L] = d[:L]
                I[i, :L] = db[:L]
            fused_np = (D, I)
        else:
            fused_np = None

        return BenchmarkArtifacts(
            recall_at_k=recall_at_k,
            auc_pr=curves.auc_pr,
            f1_max=curves.f1_max,
            max_recall_at_prec1=curves.max_recall_at_prec1,
            num_queries_total=int(len(rankings)),
            num_queries_valid=int(len(self.valid_indices)),
            curves=curves_dict,
            fused_rankings=fused_np,
        )

    def save(self, artifacts: BenchmarkArtifacts, output_dir: Path) -> None:
        out_dir = Path(output_dir).resolve()
        out_dir.mkdir(parents=True, exist_ok=True)
        metrics = {
            "config": {
                "max_window": int(self.cfg.max_window),
                "per_frame_k_used": (
                    None if self.cfg.per_frame_k_used is None else int(self.cfg.per_frame_k_used)
                ),
                "final_k": int(self.cfg.final_k),
                "recency_weighting": str(self.cfg.recency_weighting),
                "recall_threshold_m": float(self.cfg.recall_threshold_m),
            },
            "recall_at_k": artifacts.recall_at_k,
            "auc_pr": artifacts.auc_pr,
            "f1_max": artifacts.f1_max,
            "max_recall_at_prec1": artifacts.max_recall_at_prec1,
            "num_queries_total": artifacts.num_queries_total,
            "num_queries_valid": artifacts.num_queries_valid,
        }
        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

        np.savez_compressed(
            str(out_dir / "curves.npz"),
            precision=artifacts.curves["precision"],
            recall=artifacts.curves["recall"],
            thresholds=artifacts.curves["thresholds"],
            f1=artifacts.curves["f1"],
        )

        if artifacts.fused_rankings is not None:
            D, I = artifacts.fused_rankings
            np.savez_compressed(str(out_dir / "fused_rankings.npz"), distances=D, db_idx=I)
