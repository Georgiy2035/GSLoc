from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any, Callable

import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation

from opr.inference.index import FaissFlatIndex

from mmpr.pr_cache import load_pr_cache_npz
from mmpr.sequence_emulator import EmulatorConfig, emulate_sequence_fusion
from mmpr.metrics import recall_at_k_per_query, build_micro_curves_from_rankings
from mmpr.data.transforms import get_T_map_to_world
from gsloc.datasets.pr_dataset import build_valid_subset
from mmpr.pr_cache import PerFramePR


@dataclass
class SequenceBenchmarkConfig:
    q_df: pd.DataFrame
    db_df: pd.DataFrame
    frames: list[PerFramePR]
    max_window: int = 20
    per_frame_k_used: int | None = None
    final_k: int = 25
    recency_weighting: Literal["none", "linear", "exp"] = "none"
    similarity_func: Callable[[dict, dict], bool] = lambda a, b, **kwargs: False
    similarity_kwargs: dict[str, Any] | None = None


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
        self.valid_indices: list[int] = []

    # --- data loading helpers ---
    # def _load_query_xyz(self) -> np.ndarray:
    #     map_dir = (self.cfg.root_data_dir / self.cfg.map_name / "keyframe_map").resolve()
    #     traj_path = map_dir / "poses.csv"
    #     vals = np.genfromtxt(str(traj_path), delimiter=",", comments="#", dtype=np.float32)
    #     if vals.ndim == 1:
    #         vals = vals.reshape(1, -1)
    #     xyz = vals[:, 1:4].astype(np.float32, copy=False)
    #     quat = vals[:, 4:8].astype(np.float32, copy=False)
    #     T_m2w = get_T_map_to_world(self.cfg.map_name)
    #     out = np.zeros_like(xyz, dtype=np.float32)
    #     for i in range(xyz.shape[0]):
    #         Rm = Rotation.from_quat(quat[i]).as_matrix().astype(np.float32)
    #         T = np.eye(4, dtype=np.float32)
    #         T[:3, :3] = Rm
    #         T[:3, 3] = xyz[i]
    #         Tw = T_m2w @ T
    #         out[i] = Tw[:3, 3]
    #     return out

    # --- main API ---
    def run(self) -> BenchmarkArtifacts:
        """Benchmark Sequence Place Recognition (micro-averaged metrics)"""
        # Build valid subset
        # print("Building valid subset...")
        # self.valid_indices = build_valid_subset(
        #     self.cfg.q_df, 
        #     self.cfg.db_df, 
        #     self.cfg.similarity_func,
        #     **self.cfg.similarity_kwargs,
        # )
        self.valid_indices = list(range(len(self.cfg.q_df)))
        
        # Emulate sequence fusion
        cfg = EmulatorConfig(
            max_window=int(self.cfg.max_window),
            per_frame_k_used=(None if self.cfg.per_frame_k_used is None else int(self.cfg.per_frame_k_used)),
            final_k=int(self.cfg.final_k),
            recency_weighting=str(self.cfg.recency_weighting),
        )
        fused = emulate_sequence_fusion(self.cfg.frames, cfg)
        print("rankings creation started")
        # Build rankings and db_idx->xyz cache
        
        rankings: list[tuple[np.ndarray, np.ndarray]] = []
        for fused_d, fused_i in fused:
            if fused_i.size == 0:
                rankings.append((np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)))
                continue
            rankings.append((fused_d, fused_i))

        def oracle(query_id: int, db_idx: int) -> bool:
            return self.cfg.similarity_func(
                self.cfg.q_df.iloc[query_id].to_dict(), 
                self.cfg.db_df.iloc[db_idx].to_dict(), 
                **self.cfg.similarity_kwargs)

        print("ranking iteration started")
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
        
        print("recall@k calculation started")
        # Micro PR curves/metrics on valid subset
        filtered_rankings = [rankings[i] for i in self.valid_indices]
        filtered_query_ids = np.asarray(self.valid_indices, dtype=np.int64)
        print("micro curves calculation started")
        curves = build_micro_curves_from_rankings(filtered_rankings, oracle, query_ids=filtered_query_ids)

        # Prepare artifacts (curves and padded fused rankings)
        curves_dict = {
            "precision": curves.precision,
            "recall": curves.recall,
            "thresholds": curves.thresholds,
            "f1": curves.f1,
        }
        print("fused rankings preparation started")
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
