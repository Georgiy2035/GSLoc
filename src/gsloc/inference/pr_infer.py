from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Dict, Optional, Literal, Iterable, Callable

import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from opr import __version__ as opr_version
from opr.inference.index import FaissFlatIndex
from opr.inference.pipelines import PlaceRecognitionPipeline
from opr.inference.preprocessing import PointCloudMinkPreprocessor
# from opr.models.place_recognition import MinkLoc3D
# from opr.models.place_recognition.base import LateFusionModel
from opr.utils import parse_device
from PIL import Image

from mmpr.data.transforms import get_T_map_to_world
from mmpr.data.pcd import SimplePCDLoader
from mmpr.data.image import get_default_image_transform, SimpleImageLoader
from mmpr.pr_cache import PerFramePR, load_pr_cache_npz, save_pr_cache_npz
from mmpr.models import MegaLoc
from mmpr.data.multimodal import SimpleMultimodalLoader
from mmpr.seq_pr_benchmark import SequencePRBenchmarker, SequenceBenchmarkConfig
from gsloc.datasets.pr_dataset import PRDataset
from torchvision import transforms as T


from torchvision.transforms import functional as F

from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data, HeteroData

from gsloc.utils.graphs import _collate_graph_objects


def TwoModelsFrameMerge(
    frames_a: list[PerFramePR],
    frames_b: list[PerFramePR],
    *,
    distance_coef_a: float = 1.0,
    distance_coef_b: float = 1.0,
) -> list[PerFramePR]:
    """Merge two parallel ``PerFramePR`` sequences into one ranked list per frame.

    For each query index, candidates from ``frames_a`` and ``frames_b`` are
    concatenated. Distances from list ``a`` are multiplied by ``distance_coef_a``
    and distances from list ``b`` by ``distance_coef_b``. The merged frame is
    sorted by these scaled distances (ascending). ``indices`` and ``db_idx``
    rows are permuted together with ``distances``.

    Args:
        frames_a: First model's per-frame PR list.
        frames_b: Second model's per-frame PR list (same length as ``frames_a``).
        distance_coef_a: Multiplier applied to every distance from ``frames_a``.
        distance_coef_b: Multiplier applied to every distance from ``frames_b``.

    Returns:
        One ``PerFramePR`` per input frame, with ``K = K_a + K_b`` candidates each.

    Raises:
        ValueError: If the two frame lists have different lengths.
    """
    if len(frames_a) != len(frames_b):
        raise ValueError(
            f"Frame list length mismatch: {len(frames_a)} != {len(frames_b)}"
        )
    ca = float(distance_coef_a)
    cb = float(distance_coef_b)
    merged: list[PerFramePR] = []
    for fa, fb in zip(frames_a, frames_b):
        dist_a = np.asarray(fa.distances, dtype=np.float32) * ca
        dist_b = np.asarray(fb.distances, dtype=np.float32) * cb
        idx_a = np.asarray(fa.indices, dtype=np.int64, copy=False)
        idx_b = np.asarray(fb.indices, dtype=np.int64, copy=False)
        d_all = np.concatenate([dist_a, dist_b])
        i_all = np.concatenate([idx_a, idx_b])

        has_a = fa.db_idx is not None
        has_b = fb.db_idx is not None
        if has_a and has_b:
            db_a = np.asarray(fa.db_idx, dtype=np.int64, copy=False)
            db_b = np.asarray(fb.db_idx, dtype=np.int64, copy=False)
            db_all: np.ndarray | None = np.concatenate([db_a, db_b])
        elif has_a:
            ph = np.full(idx_b.shape, -1, dtype=np.int64)
            db_all = np.concatenate([np.asarray(fa.db_idx, dtype=np.int64, copy=False), ph])
        elif has_b:
            ph = np.full(idx_a.shape, -1, dtype=np.int64)
            db_all = np.concatenate([ph, np.asarray(fb.db_idx, dtype=np.int64, copy=False)])
        else:
            db_all = None

        order = np.argsort(d_all, kind="mergesort")
        d_sorted = d_all[order].astype(np.float32, copy=False)
        i_sorted = i_all[order].astype(np.int64, copy=False)
        if db_all is not None:
            db_sorted = db_all[order].astype(np.int64, copy=False)
        else:
            db_sorted = None
        merged.append(PerFramePR(indices=i_sorted, distances=d_sorted, db_idx=db_sorted))
    return merged


@dataclass
class PRInferConfig:
    df: pd.Dataframe
    root_data_dir: Path
    map_name: str
    # db_map_dir: Path
    index_dir: Path
    weights: Optional[Path] = None
    device: str = "cuda"
    per_frame_k: int = 100
    pr_quant_size: float = 0.05


class PRInferencer:
    """Run place recognition on a dataset and cache PR results.

    This class supports two modes:
    - legacy mode: pass `cfg` (old notebooks)
    - cache mode: pass `pr_pipeline` + `query_dataset`

    Cache mode features:
    - resume/merge `pr_cache.npz` when some frames are already saved
    - build and reuse `query_cache_dir/descriptors.npy`, `meta.parquet`, `schema.json`
      so future runs don't re-run the model
    - uses `batch_infer` when available; otherwise calls `infer` once per sample, passing
      the same keys as the dataloader batch with tensors sliced as ``tensor[i : i + 1]``
    """

    def __init__(
        self,
        cfg: PRInferConfig | None = None,
        *,
        pr_pipeline: PlaceRecognitionPipeline | None = None,
        query_dataset: PRDataset | None = None,
        batch_size: int = 16,
        num_workers: int = 0,
        query_cache_dir: Path | None = None,
        k: int = 100,
        device: str | torch.device | None = None,
    ) -> None:
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._query_cache_dir = query_cache_dir
        self._k_default = int(k)

        self.pr = pr_pipeline  
        self.query_dataset = query_dataset 
        self.device = parse_device(device) if device is not None else getattr(self.pr, "device", "cpu")
        self.frames: list[PerFramePR] = []
        

    def _query_cache_paths(self, query_cache_dir: Path) -> tuple[Path, Path, Path]:
        query_cache_dir = Path(query_cache_dir)
        return (
            query_cache_dir / "descriptors.npy",
            query_cache_dir / "meta.parquet",
            query_cache_dir / "schema.json",
        )

    def _load_query_descriptors_cache(self, query_cache_dir: Path) -> np.ndarray:
        desc_path, meta_path, schema_path = self._query_cache_paths(query_cache_dir)
        if not desc_path.exists() or not meta_path.exists() or not schema_path.exists():
            raise FileNotFoundError("Query cache is incomplete")
        descriptors = np.load(str(desc_path), mmap_mode="r")
        descriptors = np.asarray(descriptors, dtype=np.float32)
        return descriptors

    def _save_query_cache(self, query_cache_dir: Path, *, descriptors: np.ndarray) -> None:
        query_cache_dir = Path(query_cache_dir)
        query_cache_dir.mkdir(parents=True, exist_ok=True)

        desc_path, meta_path, schema_path = self._query_cache_paths(query_cache_dir)
        np.save(str(desc_path), descriptors.astype(np.float32, copy=False))

        if not hasattr(self.query_dataset, "save_meta_parquet"):
            raise AttributeError("query_dataset must implement save_meta_parquet(...)")
        # dataset.save_meta_parquet(meta_path=dir, meta_file=...)
        self.query_dataset.save_meta_parquet(query_cache_dir, meta_file="meta.parquet")  

        metric_enum = self.pr.index.metric() 
        metric_str = metric_enum.value if hasattr(metric_enum, "value") else str(metric_enum)

        schema = {
            "version": "1",
            "number": int(descriptors.shape[0]),
            "dim": int(descriptors.shape[1]),
            "metric": metric_str,
            "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "opr_version": getattr(opr_version, "__version__", "") if opr_version is not None else "",
        }
        schema_path.write_text(json.dumps(schema))

    def _ensure_query_descriptors_cache(
        self, 
        *, 
        query_cache_dir: Path, 
        rebuild: bool, 
        loader: DataLoader | None = None,
        infer: Callable | None = None,
        k: int = 100,
        ) -> np.ndarray:

        N = len(self.query_dataset)
        if not rebuild and query_cache_dir is not None:
            try:
                descriptors = self._load_query_descriptors_cache(query_cache_dir)
                if descriptors.shape[0] == N:
                    return descriptors
            except Exception:
                pass

        # Build descriptors (and cache them).
        loader = DataLoader(
            self.query_dataset,
            batch_size=self._batch_size,
            shuffle=False,
            num_workers=self._num_workers,
            infer=infer,
        ) if loader is None else loader

        desc_parts: list[np.ndarray] = []
        for batch in tqdm(loader, desc="Compute query descriptors"):
            desc_parts.append(infer(batch, k=k))
        descriptors = np.concatenate(desc_parts, axis=0)
        if descriptors.shape[0] != N:
            raise RuntimeError(f"Descriptor row count mismatch: {descriptors.shape[0]} != {N}")
        self._save_query_cache(query_cache_dir, descriptors=descriptors)
        return descriptors

    def _infer_pr_cache_from_descriptors(
        self, *, descriptors: np.ndarray, start_idx: int, k: int
    ) -> list[PerFramePR]:
        """Compute PerFramePR for descriptors[start_idx:] using FAISS only."""
        frames: list[PerFramePR] = []
        N = descriptors.shape[0]
        for i in tqdm(range(start_idx, N, self._batch_size), desc="Infer PR cache"):
            d_batch = descriptors[i : min(N, i + self._batch_size)]
            inds, dists = self.pr.index.search(d_batch, int(k)) 
            db_idx_flat, _db_pose_flat, _db_pc_path_flat = self.pr.index.get_meta(inds.reshape(-1)) 
            db_idx = db_idx_flat.reshape(inds.shape)
            for b in range(inds.shape[0]):
                frames.append(
                    PerFramePR(
                        indices=inds[b].astype(np.int64, copy=False),
                        distances=dists[b].astype(np.float32, copy=False),
                        db_idx=db_idx[b].astype(np.int64, copy=False),
                    )
                )
        return frames

    def run(
        self,
        *,
        k: Optional[int] = None,
        rebuild_pr_cache: bool = False,
        rebuild_query_descriptors: bool = False,
        query_cache_dir: Path | None = None,
    ) -> list[PerFramePR]:
        k_final = int(self._k_default if k is None else k)
        q_cache_dir = query_cache_dir or self._query_cache_dir

        batch_infer = getattr(self.pr, "batch_infer", None)
        one_sample_infer = getattr(self.pr, "infer", None) if not callable(batch_infer) else None
        if not callable(batch_infer) and not callable(one_sample_infer):
            raise AttributeError(
                "`pr` must define `batch_infer` or `infer` for full descriptor + PR rebuild"
            )
        # Full pass: use batch_infer to compute both descriptors and PR results.
        if callable(batch_infer):
            infer = batch_infer
            loader = DataLoader(
                self.query_dataset,
                batch_size=self._batch_size,
                shuffle=False,
                num_workers=self._num_workers,
                collate_fn=self.query_dataset.collate_fn,
            )
        else:
            assert one_sample_infer is not None
            infer = one_sample_infer
            loader = DataLoader(
                self.query_dataset,
                batch_size=1,
                shuffle=False,
                num_workers=self._num_workers,
                collate_fn=self.query_dataset.collate_fn,
            )

        N = len(self.query_dataset)
        if rebuild_pr_cache:
            existing_frames: list[PerFramePR] = []
        else:
            existing_frames = list(self.frames)
            if existing_frames and existing_frames[0].distances.shape[0] != k_final:
                existing_frames = []

        # If query descriptors are being rebuilt, we also rebuild pr_cache from scratch.
        need_full_pr_rebuild = rebuild_query_descriptors or q_cache_dir is None or (not q_cache_dir.exists())

        if need_full_pr_rebuild:
            desc_parts: list[np.ndarray] = []
            frames_full: list[PerFramePR] = []   
            
            for batch in tqdm(loader, desc="Compute descriptors + PR cache"):
                results = infer(batch, k=k_final)
                desc_parts.append(
                    np.stack([result.descriptor for result in results], axis=0).astype(np.float32, copy=False)
                )
                for result in results:
                    if result.db_idx is None:
                        raise RuntimeError("PlaceRecognitionPipeline.batch_infer must populate db_idx")
                    frames_full.append(
                        PerFramePR(
                            indices=result.indices,
                            distances=result.distances,
                            db_idx=result.db_idx,
                        )
                    )
            descriptors_all = np.concatenate(desc_parts, axis=0)
            if descriptors_all.shape[0] != N:
                raise RuntimeError(f"Descriptor row count mismatch: {descriptors_all.shape[0]} != {N}")

            if q_cache_dir is not None:
                self._save_query_cache(q_cache_dir, descriptors=descriptors_all)
            self.frames = frames_full
            return self.frames

        # Fast path: use cached descriptors and resume/merge pr_cache.
        descriptors = self._ensure_query_descriptors_cache(
            query_cache_dir=q_cache_dir, rebuild=False, loader=loader, infer=infer, k=k_final
        )
        start_idx = len(existing_frames)
        if start_idx >= N:
            self.frames = existing_frames
            return self.frames

        missing_frames = self._infer_pr_cache_from_descriptors(
            descriptors=descriptors,
            start_idx=start_idx,
            k=k_final,
        )
        self.frames = [*existing_frames, *missing_frames]
        return self.frames

    def save(
        self,
        output: Path,
        frames: Optional[list[PerFramePR]] = None,
    ) -> None:
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)

        if frames is not None:
            self.frames = list(frames)
        if not self.frames:
            raise ValueError("No frames to save. Run `run()` first or pass `frames` explicitly.")
        save_pr_cache_npz(output, self.frames)

    def load(self, path: Path) -> list[PerFramePR]:
        """Load PR cache from an npz file into `self.frames`."""
        self.frames = load_pr_cache_npz(path)
        return self.frames

    def build_recall_benchmark_report(
        self,
        *,
        database_dataset: PRDataset | None = None,
        ks: Iterable[int] | None = None,
        similarity_kwargs: dict[str, Any] | None = None,
        include_per_query: bool = False,
    ) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
        """Build Recall@N benchmark report from cached PR frames.

        A query is considered correct at `N` when at least one of the first `N`
        retrieved database candidates satisfies
        `query_dataset.similarity_check(query_sample, db_sample, **similarity_kwargs)`.

        Args:
            ks: K values to evaluate. Defaults to all K from 1 to max retrieved K.
            similarity_kwargs: Extra keyword arguments forwarded to
                `query_dataset.similarity_check(...)`.
            include_per_query: If True, also return a per-query dataframe.

        Returns:
            If `include_per_query=False`: dataframe with one row per K and columns
                `k`, `num_queries`, `num_correct`, `recall_at_k`, `recall_at_k_percent`.
            If `include_per_query=True`: tuple `(summary_df, per_query_df)`.
        """
        if not self.frames:
            raise ValueError("No frames available. Run `run()` first or load a cache before benchmarking.")
        if self.query_dataset is None:
            raise ValueError("`query_dataset` is required to compute benchmark report.")
        if database_dataset is None:
            raise ValueError("`database_dataset` is required to compute benchmark report.")
        if not hasattr(self.query_dataset, "similarity_check"):
            raise AttributeError("query_dataset must implement `similarity_check(a, b, ...)`.")

        similarity_kwargs = {} if similarity_kwargs is None else dict(similarity_kwargs)
        max_k = int(max(np.asarray(frame.indices).shape[0] for frame in self.frames))
        if max_k <= 0:
            raise ValueError("PR frames do not contain candidates to evaluate.")

        if ks is None:
            ks_eval = list(range(1, max_k + 1))
        else:
            ks_eval = sorted({int(k) for k in ks if int(k) > 0})
            if not ks_eval:
                raise ValueError("`ks` must contain at least one positive integer.")
            ks_eval = [k for k in ks_eval if k <= max_k]
            if not ks_eval:
                raise ValueError(f"All requested K values are larger than available candidates ({max_k}).")

        query_df: pd.DataFrame | None = None
        if hasattr(self.query_dataset, "df"):
            candidate_df = getattr(self.query_dataset, "df")
            if isinstance(candidate_df, pd.DataFrame):
                query_df = candidate_df

        if self.pr is None or not hasattr(self.pr, "index"):
            raise ValueError("`pr_pipeline` with a valid `index` is required to resolve database samples.")


        db_id_cache: dict[int, int] = {}

        num_queries = min(len(self.frames), len(self.query_dataset))
        first_hit_rank = np.full((num_queries,), np.iinfo(np.int32).max, dtype=np.int32) 
        per_query_rows: list[dict[str, Any]] = []

        for query_pos in tqdm(range(num_queries)):
            frame = self.frames[query_pos]
            q_sample = self.query_dataset.sample_from_position(query_pos)
            db_row_positions = np.asarray(frame.indices, dtype=np.int64)
            db_ids = np.asarray(
                frame.db_idx if frame.db_idx is not None else np.full(db_row_positions.shape, -1, dtype=np.int64),
                dtype=np.int64,
            )
            dists = np.asarray(frame.distances, dtype=np.float32)

            hit_rank: int | None = None
            for rank, row_pos in enumerate(db_row_positions.tolist(), start=1):
                db_sample = database_dataset.sample_from_position(int(row_pos))
                is_match = bool(self.query_dataset.similarity_check(q_sample, db_sample, **similarity_kwargs))
                if is_match:
                    hit_rank = rank
                    break
                if rank > max(ks_eval):
                    break

            if hit_rank is not None:
                first_hit_rank[query_pos] = int(hit_rank)

            if not include_per_query:
                continue
            
            # Build per-query dataframe
            row: dict[str, Any] = {
                "query_idx": int(query_pos),
                "num_candidates": int(db_ids.shape[0]),
                "first_hit_rank": (int(hit_rank) if hit_rank is not None else None),
                "top1_db_idx": (
                    int(db_ids[0]) if db_ids.shape[0] > 0 and int(db_ids[0]) >= 0
                    else db_id_cache.get(int(db_row_positions[0]), None) if db_row_positions.shape[0] > 0
                    else None
                ),
                "top1_distance": (float(dists[0]) if dists.shape[0] > 0 else None),
            }
           
            if query_df is not None and query_pos < len(query_df):
                for key, value in query_df.iloc[query_pos].to_dict().items():
                    row[f"query_{key}"] = value
            for k in ks_eval:
                row[f"hit_at_{k}"] = bool(hit_rank is not None and hit_rank <= k)
            per_query_rows.append(row)

        summary_rows: list[dict[str, Any]] = []
        for k in ks_eval:
            num_correct = int(np.sum(first_hit_rank <= int(k)))
            recall = float(num_correct / max(num_queries, 1))
            summary_rows.append(
                {
                    "k": int(k),
                    "num_queries": int(num_queries),
                    "num_correct": num_correct,
                    "recall_at_k": recall,
                    "recall_at_k_percent": recall * 100.0,
                }
            )

        summary_df = pd.DataFrame(summary_rows)
        if not include_per_query:
            return summary_df
        return summary_df, pd.DataFrame(per_query_rows)

    def build_sequence_recall_benchmark_report(
        self,
        *,
        database_dataset: PRDataset | None = None,
        # ks: Iterable[int] | None = None,
        similarity_kwargs: dict[str, Any] | None = None,
        seq_lengths: Iterable[int] = [5],
        per_frame_k_used: int = 25,
        save_dir: Path | None = None,
        std_mode: Literal["scene", "global"] = "global",
        scene_df_field: str | None = "scene",
    ) -> pd.DataFrame:

        """Run or reuse sequence benchmark for a map across sequence lengths."""

        if not self.frames:
            raise ValueError("No frames available. Run `run()` first or load a cache before benchmarking.")
        if self.query_dataset is None:
            raise ValueError("`query_dataset` is required to compute benchmark report.")
        if database_dataset is None:
            raise ValueError("`database_dataset` is required to compute benchmark report.")
        if not hasattr(self.query_dataset, "similarity_check"):
            raise AttributeError("query_dataset must implement `similarity_check(a, b, ...)`.")

        similarity_kwargs = {} if similarity_kwargs is None else dict(similarity_kwargs)
        
        
        all_rows: list[dict] = []
        for W in tqdm(list(seq_lengths)):
            cfg_b = SequenceBenchmarkConfig(
                q_df=self.query_dataset.df,
                db_df=database_dataset.df,
                frames=self.frames,
                max_window=int(W),
                per_frame_k_used=per_frame_k_used,
                final_k=self._k_default,
                recency_weighting="none",
                similarity_func=self.query_dataset.similarity_check,
                similarity_kwargs=similarity_kwargs,
                std_mode=std_mode,
                scene_df_field=scene_df_field,
            )

            bench = SequencePRBenchmarker(cfg_b)
            artifacts = bench.run()

            if save_dir is not None:
                w_save_dir = Path(save_dir) / f"{W:03d}-window"
                w_save_dir.mkdir(parents=True, exist_ok=True)
                bench.save(artifacts, w_save_dir)

            all_rows.append({
                "w": int(W),
                "auc_pr": float(artifacts.auc_pr),
                "f1_max": float(artifacts.f1_max),
                "recall_at_1": float(artifacts.recall_at_k.get(1, 0.0)),
                "recall_at_1_std": float(artifacts.recall_at_k_std.get(1, 0.0)),
                "recall_at_5": float(artifacts.recall_at_k.get(5, 0.0)),
                "recall_at_5_std": float(artifacts.recall_at_k_std.get(5, 0.0)),
                "recall_at_10": float(artifacts.recall_at_k.get(10, 0.0)),
                "recall_at_10_std": float(artifacts.recall_at_k_std.get(10, 0.0)),
                "recall_at_25": float(artifacts.recall_at_k.get(25, 0.0)),
                "recall_at_25_std": float(artifacts.recall_at_k_std.get(25, 0.0)),
                "num_valid": int(artifacts.num_queries_valid),
                "num_total": int(artifacts.num_queries_total),
            })

        df = pd.DataFrame(all_rows).sort_values("w").reset_index(drop=True)
        df.to_parquet(save_dir / "summaryresults.parquet")
        return df

