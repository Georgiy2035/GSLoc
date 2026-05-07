from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Any, Callable, List
from time import perf_counter

import json
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation
import random

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
    std_mode: Literal["scene", "global"] = "scene"
    scene_df_field: str | None = "scene"
    pose_df_field: str | None = "pose"
    seq_filter_kwargs: dict[str, Any] | None = None


@dataclass
class BenchmarkArtifacts:
    recall_at_k: dict[int, float]
    recall_at_k_std: dict[int, float]
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
        Ks = list(range(1, 26))
        if self.cfg.std_mode == "scene":
            scene_stats = {scene: {k: [] for k in Ks} for scene in self.cfg.q_df[self.cfg.scene_df_field].unique()}
        else:
            scene_stats = {scene: {k: [] for k in Ks} for scene in range(len(self.valid_indices) // 100)}
        
        scene_data = list(self.cfg.q_df[self.cfg.scene_df_field]) if self.cfg.scene_df_field is not None else None
        pose_data = list(self.cfg.q_df[self.cfg.pose_df_field]) if self.cfg.pose_df_field is not None else None

        # Emulate sequence fusion
        cfg = EmulatorConfig(
            max_window=int(self.cfg.max_window),
            per_frame_k_used=(None if self.cfg.per_frame_k_used is None else int(self.cfg.per_frame_k_used)),
            final_k=int(self.cfg.final_k),
            recency_weighting=str(self.cfg.recency_weighting),
        )
        t0 = perf_counter()
        fused = emulate_sequence_fusion(
            self.cfg.frames, 
            cfg, 
            scene_data=scene_data, 
            pose_data=pose_data, 
            seq_similarity_filter_mode=self.cfg.seq_filter_kwargs["seq_similarity_filter_mode"],
            seq_similarity_trans_tol_m=self.cfg.seq_filter_kwargs["seq_similarity_trans_tol_m"], 
            seq_similarity_rot_tol_deg=self.cfg.seq_filter_kwargs["seq_similarity_rot_tol_deg"]
        )
        dt = perf_counter() - t0
        self.emulate_sequence_fusion_time_mean_s = dt / len(self.cfg.frames)
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

        def oracle_query(query_id: int, db_idx: [int]) -> bool:
            q_dict = self.cfg.q_df.iloc[query_id].to_dict()
            return [self.cfg.similarity_func(
                q_dict, 
                self.cfg.db_df.iloc[db].to_dict(), 
                **self.cfg.similarity_kwargs) for db in db_idx]

        # Recall@K (K in [1..25]) on valid subset
        recalls = {k: [] for k in Ks}
        is_pos_all = []
        for qid in self.valid_indices:
            dists, db_ids = rankings[qid]
            is_pos = np.array(oracle_query(qid, db_ids), dtype=bool)
            stats = recall_at_k_per_query(db_ids, is_pos, Ks)

            random_scene = random.randint(0, len(scene_stats) - 1)
            for k in Ks:
                recalls[k].append(stats[k])
                if self.cfg.std_mode == "scene":
                    scene_stats[self.cfg.q_df.iloc[qid][self.cfg.scene_df_field]][k].append(stats[k])
                else:
                    scene_stats[random_scene][k].append(stats[k])
            is_pos_all.append(is_pos)
        
        recall_at_k = {k: float(np.mean(recalls[k]) if len(recalls[k]) > 0 else 0.0) for k in Ks}

        scene_recall_at_k = {k: {scene: float(np.mean(scene_stats[scene][k]) if len(scene_stats[scene][k]) > 0 else 0.0) for scene in scene_stats.keys()} for k in Ks}
        recall_at_k_std = {k: np.std(list(scene_recall_at_k[k].values())) for k in Ks}

        # Micro PR curves/metrics on valid subset
        filtered_rankings = [rankings[i] for i in self.valid_indices]
        filtered_query_ids = np.asarray(self.valid_indices, dtype=np.int64)
        # curves = build_micro_curves_from_rankings(filtered_rankings, oracle, query_ids=filtered_query_ids)

        # # Prepare artifacts (curves and padded fused rankings)
        curves_dict = {
            "precision": 0,#curves.precision,
            "recall": 0,#curves.recall,
            "thresholds": 0,#curves.thresholds,
            "f1": 0,#curves.f1,
        }
        max_len = max((len(r[0]) for r in rankings), default=0)
        fused_np: tuple[np.ndarray, np.ndarray] | None
        if max_len > 0:
            D = np.full((len(rankings), max_len), np.inf, dtype=np.float32)
            I = np.full((len(rankings), max_len), -1, dtype=np.int64)
            R = np.full((len(rankings), max_len), False, dtype=bool)
            for i, (d, db) in enumerate(rankings):
                L = min(len(d), max_len)
                D[i, :L] = d[:L]
                I[i, :L] = db[:L]
                R[i, :L] = is_pos_all[i][:L]
            fused_np = (D, I, R)
        else:
            fused_np = None

        return BenchmarkArtifacts(
            recall_at_k=recall_at_k,
            recall_at_k_std=recall_at_k_std,
            auc_pr=0,#curves.auc_pr,
            f1_max=0,#curves.f1_max,
            max_recall_at_prec1=0,#curves.max_recall_at_prec1,
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
                "similarity_kwargs": self.cfg.similarity_kwargs,
            },
            "recall_at_k": artifacts.recall_at_k,
            "recall_at_k_std": artifacts.recall_at_k_std,
            "auc_pr": artifacts.auc_pr,
            "f1_max": artifacts.f1_max,
            "sequence_fusion_time_mean_s": self.emulate_sequence_fusion_time_mean_s,
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
            D, I, R = artifacts.fused_rankings
            np.savez_compressed(str(out_dir / "fused_rankings.npz"), distances=D, db_idx=I, is_pos=R)
