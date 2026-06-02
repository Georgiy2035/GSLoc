from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, Optional, Literal, Iterable, Callable

from gsloc.datasets.replica import R
import numpy as np
import torch
from tqdm import tqdm
import pandas as pd
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
import json

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

from mmpr.inference.pipelines import PlaceRecognitionRerankPipeline


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
        k: int = 100,
        device: str | torch.device | None = None,
        time_test: bool = False,
    ) -> None:
        self._batch_size = int(batch_size)
        self._num_workers = int(num_workers)
        self._k_default = int(k)

        self.pr = pr_pipeline  
        self.query_dataset = query_dataset 
        self.device = parse_device(device) if device is not None else getattr(self.pr, "device", "cpu")
        self.frames: list[PerFramePR] = []
        self.time_test = time_test
        self.rerank_mode = getattr(self.pr, "rerank_mode", None) is not None

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

    def _ensure_query_descriptors_cache(self, *, query_cache_dir: Path) -> np.ndarray:
        N = len(self.query_dataset)
        if query_cache_dir is not None:
            descriptors = self._load_query_descriptors_cache(query_cache_dir)
            if descriptors.shape[0] == N:
                return descriptors
        assert False, "Descriptors cache is incomplete"

    def run(
        self,
        *,
        k: Optional[int] = None,
        rebuild_pr_cache: bool = False,
        rebuild_query_descriptors: bool = False,
        query_cache_dir: Path | None = None,
        rerank_query_cache_dir: Path | None = None,
        database_dataset: PRDataset | None = None, # for self-rerank mode
        **kwargs: Any,
    ) -> list[PerFramePR]:
        k_final = int(self._k_default if k is None else k)
        q_cache_dir = query_cache_dir
        rq_cache_dir = rerank_query_cache_dir if self.rerank_mode and rerank_query_cache_dir is not None else None
        N = len(self.query_dataset)

        if not rebuild_pr_cache and len(self.frames) > 0:
            return self.frames

        #######INFER FUNC AND LOADER CREATION########################
        batch_infer = getattr(self.pr, "batch_infer", None)
        one_sample_infer = getattr(self.pr, "infer", None)
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

        #########TIME TEST#################
        N_test = 100
        self.inference_time = 0
        count = 0

        if self.time_test:
            time_infer = one_sample_infer
            time_test_dataloader = DataLoader(
                        self.query_dataset, 
                        batch_size=1, 
                        shuffle=True, 
                        num_workers=self._num_workers, 
                        collate_fn=self.query_dataset.collate_fn
                    )
            for batch in tqdm(time_test_dataloader, desc="Time test", total=N_test):
                count += 1
                if count > N_test:
                    break
                start_time = perf_counter()
                results = time_infer(batch, k=k_final, dataset=database_dataset)
                end_time = perf_counter()
                self.inference_time += end_time - start_time
            self.inference_time /= N_test
        
        self.extract_time_mean_s = self.pr.extract_time_mean_s
        self.index_search_time_mean_s = self.pr.index_search_time_mean_s
        self.extract_calls = self.pr._extract_calls
        self.index_search_calls = self.pr._index_search_calls

        if self.rerank_mode:
            self.rerank_extract_time_mean_s = self.pr.rerank_extract_time_mean_s
            self.rerank_index_search_time_mean_s = self.pr.rerank_index_search_time_mean_s
            self.rerank_extract_calls = self.pr._rerank_extract_calls
        else:
            self.rerank_extract_time_mean_s = 0
            self.rerank_index_search_time_mean_s = 0
            self.rerank_extract_calls = 0

        #######FULL QUERY DESCRIPTORS AND PR CACHE CALCULATION########################
        # If query descriptors are being rebuilt, we also rebuild pr_cache from scratch.
        need_full_pr_rebuild = rebuild_query_descriptors or q_cache_dir is None or (not q_cache_dir.exists())
        using_unsavable_rerank = self.rerank_mode and self.pr.rerank_index is None
        need_descriptors_save = q_cache_dir is not None and (not q_cache_dir.exists() or rebuild_query_descriptors)
        need_rerank_descriptors_save = self.rerank_mode and rq_cache_dir is not None and (not rq_cache_dir.exists() or rebuild_query_descriptors)

        print("q_cache_dir: ", q_cache_dir)
        print("rq_cache_dir: ", rq_cache_dir)
        print("need_full_pr_rebuild: ", need_full_pr_rebuild)
        print("using_unsavable_rerank: ", using_unsavable_rerank)
        print("need_descriptors_save: ", need_descriptors_save)
        print("need_rerank_descriptors_save: ", need_rerank_descriptors_save)
        
        frames_full: list[PerFramePR] = []
        if need_full_pr_rebuild or using_unsavable_rerank or need_rerank_descriptors_save:
            desc_parts: list[np.ndarray] = []
            rerank_desc_parts: list[np.ndarray] = []
            
            for batch in tqdm(loader, desc="Compute descriptors + PR cache"):
                
                results = infer(batch, k=k_final, dataset=database_dataset)
                
                if need_descriptors_save:
                    desc_parts.append(
                        np.stack([result.descriptor for result in results], axis=0).astype(np.float32, copy=False)
                    )
                if need_rerank_descriptors_save:
                    assert getattr(results[0], "rerank_descriptor", None) is not None, "rerank_descriptor is required"
                    rerank_desc_parts.append(
                        np.stack([result.rerank_descriptor for result in results], axis=0).astype(np.float32, copy=False)
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

            if need_descriptors_save:
                descriptors_all = np.concatenate(desc_parts, axis=0)          
                if descriptors_all.shape[0] != N:
                    raise RuntimeError(f"Descriptor row count mismatch: {descriptors_all.shape[0]} != {N}")
                self._save_query_cache(q_cache_dir, descriptors=descriptors_all)
                
            if need_rerank_descriptors_save:
                rerank_descriptors_all = np.concatenate(rerank_desc_parts, axis=0)
                self._save_query_cache(rq_cache_dir, descriptors=rerank_descriptors_all)
            
            self.frames = frames_full
            return self.frames

        #######USE CACHED DESCRIPTORS########################
        # Fast path: use cached descriptors.
        print("Using cached descriptors for query: loading from ", q_cache_dir)
        descriptors = self._ensure_query_descriptors_cache(
            query_cache_dir=q_cache_dir
        )
        rerank_descriptors = None
        if self.rerank_mode and rq_cache_dir is not None:
            rerank_descriptors = self._ensure_query_descriptors_cache(
                query_cache_dir=rq_cache_dir
            )
        using_rerank = rerank_descriptors is not None

        print("Descriptors loaded: ", descriptors.shape, rerank_descriptors.shape, "getting results")
        iter_d = zip(descriptors, rerank_descriptors) if using_rerank else descriptors
        for ds in tqdm(iter_d, desc="retrieval"):
            results = self.pr._search_descriptors(np.array([ds[0]]), np.array([ds[1]]), k_final, dataset=database_dataset) \
                if using_rerank else self.pr._search_descriptors(np.array([ds]), k_final)
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
        print("Results got: ", len(frames_full))

        self.frames = frames_full
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
        if self.time_test:
            self._save_time_data(output.parent / "time.json")

    def _save_time_data(self, time_path: Path) -> None:
        time_schema = {
            "full_inference_time": self.inference_time,
            "extract_time_mean_s": self.extract_time_mean_s,
            "rerank_extract_time_mean_s": self.rerank_extract_time_mean_s,
            "extract_time_calls": self.extract_calls,
            "rerank_extract_time_calls": self.rerank_extract_calls,
            "index_search_time_mean_s": self.index_search_time_mean_s,
            "rerank_index_search_time_mean_s": self.rerank_index_search_time_mean_s,
            "index_search_time_calls": self.index_search_calls,
        }
        time_path.write_text(json.dumps(time_schema))

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

        # if self.pr is None or not hasattr(self.pr, "index"):
        #     raise ValueError("`pr_pipeline` with a valid `index` is required to resolve database samples.")


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

    @staticmethod
    def _frame_label_from_path(path: Any) -> str:
        if path is None or (isinstance(path, float) and pd.isna(path)):
            return ""
        name = Path(str(path)).name
        if name.endswith(".color.jpg"):
            return name[: -len(".color.jpg")]
        return Path(str(path)).stem

    def _write_query_scene_top1_match_files(
        self,
        *,
        q_df: pd.DataFrame,
        db_df: pd.DataFrame,
        fused_rankings: tuple[np.ndarray, np.ndarray, np.ndarray],
        output_dir: Path,
        scene_df_field: str,
    ) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        _, db_row_ids, _ = fused_rankings
        scene_lines: dict[str, list[str]] = {}

        for qid in range(len(q_df)):
            if db_row_ids.shape[1] == 0 or int(db_row_ids[qid, 0]) < 0:
                continue

            query_row = q_df.iloc[qid]
            db_row = db_df.iloc[int(db_row_ids[qid, 0])]
            scene_id = str(query_row[scene_df_field])
            line = (
                f"{self._frame_label_from_path(query_row.get('image_path'))} "
                f"{self._frame_label_from_path(db_row.get('image_path'))} "
                f"{db_row[scene_df_field]}"
            )
            scene_lines.setdefault(scene_id, []).append(line)

        for scene_id, lines in scene_lines.items():
            (output_dir / f"{scene_id}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

    def build_sequence_recall_benchmark_report(
        self,
        *,
        database_dataset: PRDataset | None = None,
        # ks: Iterable[int] | None = None,
        similarity_kwargs: dict[str, Any] | None = None,
        seq_lengths: Iterable[int] = [5],
        per_frame_k_used: int = 25,
        final_k: int | None = 25,
        save_dir: Path | None = None,
        std_mode: Literal["scene", "global"] = "global",
        scene_df_field: str | None = "scene",
        pose_df_field: str | None = "pose",
        seq_filter_kwargs: dict[str, Any] | None = None,
        recall_at_k: Iterable[int] | None = None,
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
                final_k=final_k if final_k is not None else self._k_default,
                recency_weighting="none",
                similarity_func=self.query_dataset.similarity_check,
                similarity_kwargs=similarity_kwargs,
                std_mode=std_mode,
                scene_df_field=scene_df_field,
                pose_df_field=pose_df_field,
                seq_filter_kwargs=seq_filter_kwargs,
            )

            bench = SequencePRBenchmarker(cfg_b)
            artifacts = bench.run()

            if save_dir is not None:
                w_save_dir = Path(save_dir) / f"{W:03d}-window"
                w_save_dir.mkdir(parents=True, exist_ok=True)
                bench.save(artifacts, w_save_dir)
                
                ##### WRITE TOP 1 MATCH FILES #####
                # if (
                #     int(W) == 1
                #     and artifacts.fused_rankings is not None
                #     and scene_df_field is not None
                # ):
                #     self._write_query_scene_top1_match_files(
                #         q_df=self.query_dataset.df,
                #         db_df=database_dataset.df,
                #         fused_rankings=artifacts.fused_rankings,
                #         output_dir=w_save_dir / "top1_match_files",
                #         scene_df_field=scene_df_field,
                #     )

            all_rows.append({
                "w": int(W),
                "auc_pr": float(artifacts.auc_pr),
                "f1_max": float(artifacts.f1_max),
                **{f"recall_at_{k}": float(artifacts.recall_at_k.get(k, 0.0)) for k in recall_at_k},
                **{f"recall_at_{k}_std": float(artifacts.recall_at_k_std.get(k, 0.0)) for k in recall_at_k},
                "num_valid": int(artifacts.num_queries_valid),
                "num_total": int(artifacts.num_queries_total),
            })

        df = pd.DataFrame(all_rows).sort_values("w").reset_index(drop=True)
        df.to_parquet(save_dir / "summaryresults.parquet")
        return df

# class PRRerankInferencer(PRInferencer):
#     """Run place recognition on a dataset and cache PR results.

#     This class supports two modes:
#     - legacy mode: pass `cfg` (old notebooks)
#     - cache mode: pass `pr_pipeline` + `query_dataset`

#     Cache mode features:
#     - resume/merge `pr_cache.npz` when some frames are already saved
#     - build and reuse `query_cache_dir/descriptors.npy`, `meta.parquet`, `schema.json`
#       so future runs don't re-run the model
#     - uses `batch_infer` when available; otherwise calls `infer` once per sample, passing
#     the same keys as the dataloader batch with tensors sliced as ``tensor[i : i + 1]``
#     """

#     def __init__(
#         self,
#         cfg: PRInferConfig | None = None,
#         *,
#         pr_rerank_pipeline: PlaceRecognitionRerankPipeline | None = None,
#         query_dataset: PRDataset | None = None,
#         batch_size: int = 16,
#         num_workers: int = 0,
#         query_cache_dir: Path | None = None,
#         k: int = 100,
#         device: str | torch.device | None = None,
#         time_test: bool = False,
#     ) -> None:
#         super().__init__(
#             cfg, 
#             pr_pipeline=pr_rerank_pipeline, 
#             query_dataset=query_dataset, 
#             batch_size=batch_size, 
#             num_workers=num_workers, 
#             query_cache_dir=query_cache_dir, 
#             k=k, 
#             device=device,
#             time_test=time_test
#         )
#         self.pr = pr_rerank_pipeline

    

#     def run(
#         self,
#         *,
#         k: Optional[int] = None,
#         rebuild_pr_cache: bool = False,
#         rebuild_query_descriptors: bool = False,
#         query_cache_dir: str | Path | None = None,
#         rerank_query_cache_dir: Path | None = None,
#         **kwargs: Any,
#     ) -> list[PerFramePR]:
#         k_final = int(self._k_default if k is None else k)
#         q_cache_dir = Path(query_cache_dir) or Path(self._query_cache_dir)
#         N = len(self.query_dataset)
#         rq_cache_dir = Path(rerank_query_cache_dir)
        
#         if not rebuild_pr_cache and len(self.frames) > 0:
#             return self.frames

#         #######INFER FUNC AND LOADER CREATION########################
#         batch_infer = getattr(self.pr, "batch_infer", None)
#         one_sample_infer = getattr(self.pr, "infer", None)
#         if not callable(batch_infer) and not callable(one_sample_infer):
#             raise AttributeError(
#                 "`pr` must define `batch_infer` or `infer` for full descriptor + PR rebuild"
#             )
#         # Full pass: use batch_infer to compute both descriptors and PR results.
#         if callable(batch_infer):
#             infer = batch_infer
#             loader = DataLoader(
#                 self.query_dataset,
#                 batch_size=self._batch_size,
#                 shuffle=False,
#                 num_workers=self._num_workers,
#                 collate_fn=self.query_dataset.collate_fn,
#             )
#         else:
#             assert one_sample_infer is not None
#             infer = one_sample_infer
#             loader = DataLoader(
#                 self.query_dataset,
#                 batch_size=1,
#                 shuffle=False,
#                 num_workers=self._num_workers,
#                 collate_fn=self.query_dataset.collate_fn,
#             )


#         #########TIME TEST#################
#         N_test = 100
#         self.inference_time = 0
#         count = 0

#         if self.time_test:
#             time_infer = one_sample_infer
#             time_test_dataloader = DataLoader(
#                         self.query_dataset, 
#                         batch_size=1, 
#                         shuffle=True, 
#                         num_workers=self._num_workers, 
#                         collate_fn=self.query_dataset.collate_fn
#                     )
#             for batch in tqdm(time_test_dataloader, desc="Time test", total=N_test):
#                 count += 1
#                 if count > N_test:
#                     break
#                 start_time = perf_counter()
#                 results = time_infer(batch, k=k_final)
#                 end_time = perf_counter()
#                 self.inference_time += end_time - start_time
#             self.inference_time /= N_test

#         self.extract1_time_mean_s = self.pr.extract1_time_mean_s
#         self.extract2_time_mean_s = self.pr.extract2_time_mean_s
#         self.extract_calls = self.pr._extract_calls
#         self.index1_search_time_mean_s = self.pr.index1_search_time_mean_s
#         self.index2_search_time_mean_s = self.pr.index2_search_time_mean_s
#         self.index1_search_calls = self.pr._index1_search_calls
#         self.index2_search_calls = self.pr._index2_search_calls
            
#         #######FULL QUERY DESCRIPTORS AND PR CACHE CALCULATION########################
#         # If query descriptors are being rebuilt, we also rebuild pr_cache from scratch.
#         need_full_pr_rebuild = rebuild_query_descriptors or q_cache_dir is None or (not q_cache_dir.exists())
#         if need_full_pr_rebuild:
#             desc_parts: list[np.ndarray] = []
#             desc_parts2: list[np.ndarray] = []
#             frames_full: list[PerFramePR] = []   
            
#             for batch in tqdm(loader, desc="Compute descriptors + PR cache"):
#                 results = infer(batch, k=k_final)
#                 desc_parts.append(np.stack([result.descriptor for result in results], axis=0).astype(np.float32, copy=False))
#                 desc_parts2.append(np.stack([result.descriptor2 for result in results], axis=0).astype(np.float32, copy=False))
#                 for result in results:
#                     if result.db_idx is None:
#                         raise RuntimeError("PlaceRecognitionPipeline.batch_infer must populate db_idx")
#                     frames_full.append(
#                         PerFramePR(
#                             indices=result.indices,
#                             distances=result.distances,
#                             db_idx=result.db_idx,
#                         )
#                     )
#             descriptors_all = np.concatenate(desc_parts, axis=0)
#             descriptors_all2 = np.concatenate(desc_parts2, axis=0)
#             if descriptors_all.shape[0] != N:
#                 raise RuntimeError(f"Descriptor row count mismatch: {descriptors_all.shape[0]} != {N}")

#             if q_cache_dir is not None:
#                 self._save_query_cache(q_cache_dir, descriptors=descriptors_all)    
#                 self._save_query_cache(rq_cache_dir, descriptors=descriptors_all2)
#             self.frames = frames_full
#             return self.frames

#         #######USE CACHED DESCRIPTORS########################
#         # Fast path: use cached descriptors.
#         print("Using cached descriptors for query and rerank: loading from ", q_cache_dir, rq_cache_dir)
#         descriptors = self._ensure_query_descriptors_cache(
#             query_cache_dir=q_cache_dir
#         )
#         descriptors2 = self._ensure_query_descriptors_cache(
#             query_cache_dir=rq_cache_dir
#         )
#         frames_full: list[PerFramePR] = []
#         print("Descriptors loaded: ", descriptors.shape, descriptors2.shape, "getting results")
#         for d1, d2 in tqdm(zip(descriptors, descriptors2), desc="retrieval"):
#             results = self.pr._search_descriptors(np.array([d1]), np.array([d2]), k_final)
#             for result in results:
#                 if result.db_idx is None:
#                     raise RuntimeError("PlaceRecognitionPipeline.batch_infer must populate db_idx")
#                 frames_full.append(
#                     PerFramePR(
#                         indices=result.indices,
#                         distances=result.distances,
#                         db_idx=result.db_idx,
#                     )
#                 )
#         print("Results got: ", len(frames_full))

#         self.frames = frames_full
#         return self.frames

#     def _save_time_data(self, time_path: Path) -> None:
#         time_schema = {
#             "full_inference_time": self.inference_time,
#             "extract1_time_mean_s": self.extract1_time_mean_s,
#             "extract2_time_mean_s": self.extract2_time_mean_s,
#             "extract_time_calls": self.extract_calls,
#             "index1_search_time_mean_s": self.index1_search_time_mean_s,
#             "index2_search_time_mean_s": self.index2_search_time_mean_s,
#             "index1_search_time_calls": self.index1_search_calls,
#             "index2_search_time_calls": self.index2_search_calls,
#         }
#         time_path.write_text(json.dumps(time_schema))