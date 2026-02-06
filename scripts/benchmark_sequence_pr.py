#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Literal

import json
import numpy as np

from opr.inference.index import FaissFlatIndex

from mmpr.pr_cache import load_pr_cache_npz
from mmpr.sequence_emulator import EmulatorConfig, emulate_sequence_fusion
from mmpr.metrics import recall_at_k_per_query, build_micro_curves_from_rankings
from mmpr.data.transforms import get_T_map_to_world
from scipy.spatial.transform import Rotation


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Benchmark Sequence Place Recognition (micro-averaged metrics)")
    p.add_argument("--db-index-dir", type=Path, required=True, help="DB FAISS index dir (to access db_idx if needed)")
    p.add_argument("--cache", type=Path, required=True, help="Input PR cache npz per map")
    p.add_argument("--root-data-dir", type=Path, required=True, help="Dataset root (for GT poses)")
    p.add_argument("--map", dest="map_name", type=str, required=True, help="Map name (e.g., map2)")
    p.add_argument("--max-window", type=int, default=20)
    p.add_argument("--per-frame-k-used", type=int, default=None)
    p.add_argument("--final-k", type=int, default=25)
    p.add_argument("--recency-weighting", type=str, default="none", choices=["none", "linear", "exp"])
    p.add_argument("--recall-threshold-m", type=float, default=5.0, help="Distance threshold for match (meters, PR-space)")
    p.add_argument("--output-dir", type=Path, required=True)
    return p.parse_args()


def main() -> None:
    a = parse_args()
    out_dir = a.output_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load PR cache and DB index (for db poses if needed)
    frames = load_pr_cache_npz(a.cache)
    index = FaissFlatIndex.load(str(a.db_index_dir))

    # Build GT-based spatial oracle: query_xyz from poses.csv (map frame transformed to world/map1),
    # and db_idx->db_xyz from index metadata for candidates encountered.
    def _load_query_xyz(root_data_dir: Path, map_name: str) -> np.ndarray:
        map_dir = (root_data_dir / map_name / "keyframe_map").resolve()
        traj_path = map_dir / "poses.csv"
        vals = np.genfromtxt(str(traj_path), delimiter=",", comments="#", dtype=np.float32)
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        # columns: timestamp, x, y, z, qx, qy, qz, qw
        xyz = vals[:, 1:4].astype(np.float32, copy=False)
        quat = vals[:, 4:8].astype(np.float32, copy=False)
        T_m2w = get_T_map_to_world(map_name)
        out = np.zeros_like(xyz, dtype=np.float32)
        for i in range(xyz.shape[0]):
            Rm = Rotation.from_quat(quat[i]).as_matrix().astype(np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rm
            T[:3, 3] = xyz[i]
            Tw = T_m2w @ T
            out[i] = Tw[:3, 3]
        return out

    query_xyz = _load_query_xyz(a.root_data_dir, a.map_name)
    db_xyz_by_db_idx: dict[int, np.ndarray] = {}

    # Determine which queries are valid (have at least one DB pose within threshold)
    # Use all DB poses from the index when available; otherwise fall back to treating all queries as valid.
    valid_queries: list[bool] = []
    thr = float(a.recall_threshold_m)
    db_all_xyz = None
    try:
        db_all_xyz = index._db_pose[:, :3]  # type: ignore[attr-defined]
    except Exception:
        db_all_xyz = None
    if isinstance(db_all_xyz, np.ndarray):
        for q in query_xyz:
            d = np.linalg.norm(db_all_xyz - q[None, :], axis=1)
            valid_queries.append(bool(np.any(d < thr)))
    else:
        valid_queries = [True] * query_xyz.shape[0]

    # Emulate sequence fusion once for given config
    cfg = EmulatorConfig(max_window=int(a.max_window), per_frame_k_used=(None if a.per_frame_k_used is None else int(a.per_frame_k_used)), final_k=int(a.final_k), recency_weighting=str(a.recency_weighting))
    fused = emulate_sequence_fusion(frames, cfg)

    # Rankings for micro-PR: distances and db_idx; also build db_idx->xyz mapping as we go
    rankings: list[tuple[np.ndarray, np.ndarray]] = []
    for (fused_d, fused_i) in fused:
        # Map internal row ids to db_idx using index
        if fused_i.size == 0:
            rankings.append((np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.int64)))
            continue
        db_idx, db_pose, _db_pc = index.get_meta(fused_i)
        # Cache xyz per db_idx for oracle
        for j in range(db_idx.shape[0]):
            did = int(db_idx[j])
            if did not in db_xyz_by_db_idx:
                db_xyz_by_db_idx[did] = np.asarray(db_pose[j][:3], dtype=np.float32)
        rankings.append((fused_d, db_idx))


    def oracle(query_id: int, db_idx: int) -> bool:
        if not (0 <= query_id < query_xyz.shape[0]):
            return False
        q = query_xyz[int(query_id)]
        d = db_xyz_by_db_idx.get(int(db_idx))
        if d is None:
            return False
        return float(np.linalg.norm(q - d)) < thr

    # Recall@K for K in [1..25] on the valid subset only
    Ks = list(range(1, 26))
    recalls = {k: [] for k in Ks}
    valid_indices = [i for i, v in enumerate(valid_queries) if v]
    for qid in valid_indices:
        dists, db_ids = rankings[qid]
        is_pos = np.array([oracle(qid, int(db)) for db in db_ids], dtype=bool)
        stats = recall_at_k_per_query(db_ids, is_pos, Ks)
        for k in Ks:
            recalls[k].append(stats[k])
    recall_at_k = {k: float(np.mean(recalls[k]) if len(recalls[k]) > 0 else 0.0) for k in Ks}

    # Micro PR curves and metrics
    filtered_rankings = [rankings[i] for i in valid_indices]
    filtered_query_ids = np.asarray(valid_indices, dtype=np.int64)
    curves = build_micro_curves_from_rankings(filtered_rankings, oracle, query_ids=filtered_query_ids)

    # Save artifacts
    # 1) metrics.json
    metrics = {
        "config": {
            "max_window": cfg.max_window,
            "per_frame_k_used": cfg.per_frame_k_used,
            "final_k": cfg.final_k,
            "recency_weighting": cfg.recency_weighting,
        },
        "recall_at_k": recall_at_k,
        "auc_pr": curves.auc_pr,
        "f1_max": curves.f1_max,
        "max_recall_at_prec1": curves.max_recall_at_prec1,
        "num_queries_total": int(len(rankings)),
        "num_queries_valid": int(len(valid_indices)),
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # 2) curves.npz
    np.savez_compressed(
        str(out_dir / "curves.npz"),
        precision=curves.precision,
        recall=curves.recall,
        thresholds=curves.thresholds,
        f1=curves.f1,
    )

    # 3) fused_rankings.npz (for reproducibility)
    # Store variable-length by padding to max length with -1 and +inf
    max_len = max((len(r[0]) for r in rankings), default=0)
    if max_len > 0:
        D = np.full((len(rankings), max_len), np.inf, dtype=np.float32)
        I = np.full((len(rankings), max_len), -1, dtype=np.int64)
        for i, (d, db) in enumerate(rankings):
            L = min(len(d), max_len)
            D[i, :L] = d[:L]
            I[i, :L] = db[:L]
        np.savez_compressed(str(out_dir / "fused_rankings.npz"), distances=D, db_idx=I)

    print(f"Saved benchmark artifacts to {out_dir}")


if __name__ == "__main__":
    main()


