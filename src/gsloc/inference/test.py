import itertools
import shutil
from pathlib import Path
import json

from dataclasses import dataclass, field
import faiss
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import plotly.graph_objects as go
from typing import Literal
import torch.nn as nn

from torchvision import transforms as T
from opr.datasets.itlp import ITLPCampus
#from opr.models.place_recognition import MinkLoc3D
from mmpr.inference import PlaceRecognitionPipeline, FaissFlatIndex, SequencePlaceRecognitionPipeline, PlaceRecognitionRerankPipeline

from gsloc.inference.pr_infer import PRInferencer
from gsloc.models import opr_graph_extention as network
# from opr.pipelines.place_recognition import PlaceRecognitionPipeline

from mmpr.models import MegaLoc
from gsloc.datasets import ThreeRScan
from gsloc.datasets.pr_dataset import PRDataset

@dataclass
class TestConfig:
    ########PATHS#################
    dataset_path: Path
    test_path: Path
    index_path: Path
    rerank_index_path: Path
    query_cache_path: Path
    rerank_query_cache_path: Path
    bench_report_path: Path
    frames_path: Path

    ########DATASET CONFIG#################
    dataset_class: PRDataset
    filter_kwargs: dict = field(default_factory=lambda: {
        "similarity_filter_mode": "none",
        "similarity_trans_tol_m": 2,
        "similarity_rot_tol_deg": 90
    })
    scene_list_path: Path | None = None
    query_list_path: Path | None = None
    room_json_path: Path | None = None
    image_transform_fn: T.Compose | None = None
    graph_path: Path | None = None #graph dir with .pt files in scenes (depends on graph source - GT, FROSS or VLMGD)
    edge_normalizer_path: Path | None = None
    graph_feat_dim: int | None = None
    graph_edge_attr_dim: int | None = None
    graph_rotate: bool | None = None
    modality: tuple[Literal["image"], Literal["graph"]] = ("image", "graph")

    ########MODEL RUN CONFIG#################
    model: nn.Module | None = None
    rerank_model: nn.Module | None = None
    model_self_rerank_flag: bool = False #if True, the reranking is performed by the same model
    rerank_descriptor_save_flag: bool = True #if True, the rerank descriptors are saved
    device: str = "cuda"
    num_workers: int = 4
    batch_size: int = 16
    time_test: bool = False
    
    ########RERANK CONFIG#################
    rerank_k: int = 500 #number of closest frames returned by first model to rerank
    
    ########SEQUENCE CONFIG#################
    per_frame_k_used: int = 25
    final_k: int = 25
    seq_lengths: list[int] = field(default_factory=lambda: [1, 2, 3, 5, 7, 10, 15, 20, 25, 30, 35])
    seq_filter_kwargs: dict = field(default_factory=lambda: {
        "seq_similarity_filter_mode": "none",
        "seq_similarity_trans_tol_m": 2,
        "seq_similarity_rot_tol_deg": 90
    })
    ########BENCHMARK CONFIG#################
    recall_at_k: list[int] = field(default_factory=lambda: [1, 5, 10, 25]) #@k values for recall calculation
    similarity_kwargs: dict = field(default_factory=lambda: {
        "mode": "room",
    })
    std_mode: Literal["scene", "global"] = "global"
    scene_df_field: str = "scene" #for std calculation
    pose_df_field: str = "pose" #for pose calculation
    


class Test:
    def __init__(self, cfg: TestConfig):
        self.cfg = cfg

        cfg.index_path = cfg.test_path / "index" if cfg.index_path is None else cfg.index_path
        # cfg.rerank_index_path = cfg.test_path / "rerank_index" if cfg.rerank_index_path is None else cfg.rerank_index_path
        # cfg.query_cache_path = cfg.test_path / "query_cache" if cfg.query_cache_path is None else cfg.query_cache_path
        # cfg.rerank_query_cache_path = cfg.test_path / "rerank_query_cache" if cfg.rerank_query_cache_path is None else cfg.rerank_query_cache_path
        cfg.bench_report_path = cfg.test_path / "bench_report" if cfg.bench_report_path is None else cfg.bench_report_path  
        cfg.frames_path = cfg.bench_report_path / "frames" if cfg.frames_path is None else cfg.frames_path
        
        
        self.database_dataset = cfg.dataset_class(
            dataset_root=cfg.dataset_path,
            meta_path=cfg.index_path,
            rebuild_meta=False,  # meta.parquet already built
            image_transform=cfg.image_transform_fn,
            save_meta=False,
            scene_filter_mode="listed",
            scene_list_path=cfg.scene_list_path,
            room_json_path=cfg.room_json_path,
            graph_feat_dim = cfg.graph_feat_dim,
            graph_edge_attr_dim = cfg.graph_edge_attr_dim,
            graph_rotate = cfg.graph_rotate,
            graph_path=cfg.graph_path,
            edge_normalizer_path=cfg.edge_normalizer_path,
            modality=cfg.modality,
            **cfg.filter_kwargs
        )

        self.query_dataset = cfg.dataset_class(
            dataset_root=cfg.dataset_path,
            meta_path=cfg.query_cache_path,
            rebuild_meta=False,
            image_transform=cfg.image_transform_fn,
            scene_filter_mode="same_room_excluding_listed" if cfg.query_list_path is None else "listed",
            scene_list_path=cfg.scene_list_path if cfg.query_list_path is None else cfg.query_list_path,
            room_json_path=cfg.room_json_path,
            graph_feat_dim = cfg.graph_feat_dim,
            graph_edge_attr_dim = cfg.graph_edge_attr_dim,
            graph_rotate = cfg.graph_rotate,
            graph_path=cfg.graph_path,
            edge_normalizer_path=cfg.edge_normalizer_path,
            modality=cfg.modality,
        )

        self.index = FaissFlatIndex.generate(
            directory=cfg.index_path,
            dataset=self.database_dataset,
            dataloader=None,
            model=cfg.model,
            rebuild_meta=False,
            rebuild_descriptors=False,
            batch_size = cfg.batch_size,
            num_workers = cfg.num_workers,
            shuffle = False,
            metric = "l2", # can be also "ip" - inner product
            version = 1
        )

        if cfg.model_self_rerank_flag:

            self.rerank_index = None
            if cfg.rerank_descriptor_save_flag:
                self.rerank_index = FaissFlatIndex.generate(
                    directory=cfg.rerank_index_path,
                    dataset=self.database_dataset,
                    dataloader=None,
                    model=cfg.model,
                    descriptor_key="rerank_descriptor",
                    rebuild_meta=False,
                    rebuild_descriptors=False,
                    batch_size = cfg.batch_size,
                    num_workers = cfg.num_workers,
                    shuffle = False,
                    metric = "l2", # can be also "ip" - inner product
                    version = 1
                )

            self.pipeline = PlaceRecognitionRerankPipeline(
                index=self.index,
                rerank_index=self.rerank_index,
                model=cfg.model,
                rerank_model=None,
                device=cfg.device,
            )


        elif cfg.rerank_model is None:

            self.pipeline = PlaceRecognitionPipeline(
                index=self.index,
                model=cfg.model,
                device=cfg.device
            )

        else:

            self.rerank_index = FaissFlatIndex.generate(
                directory=cfg.rerank_index_path,
                dataset=self.database_dataset,
                dataloader=None,
                model=cfg.rerank_model,
                rebuild_meta=False,
                rebuild_descriptors=False,
                batch_size = cfg.batch_size,
                num_workers = cfg.num_workers,
                shuffle = False,
                metric = "l2", # can be also "ip" - inner product
                version = 1
            )

            self.pipeline = PlaceRecognitionRerankPipeline(
                index=self.index,
                rerank_index=self.rerank_index,
                model=cfg.model,
                rerank_model=cfg.rerank_model,
                device=cfg.device,
            )
            
        self.inferencer = PRInferencer(
                pr_pipeline=self.pipeline,
                query_dataset=self.query_dataset,
                batch_size=cfg.batch_size,
                num_workers=cfg.num_workers,
                k=cfg.rerank_k,
                device=cfg.device,
                time_test = cfg.time_test
            )


    def run(self):
        if self.cfg.frames_path.exists():
            print(f"Loading frames from {self.cfg.frames_path}")
            frames = self.inferencer.load(self.cfg.frames_path)
        else:
            print(f"Running inference and saving frames to {self.cfg.frames_path}")
            frames = self.inferencer.run(
                query_cache_dir=self.cfg.query_cache_path, 
                rerank_query_cache_dir=self.cfg.rerank_query_cache_path, 
                rebuild_query_descriptors=False,
                rebuild_pr_cache=False,
                database_dataset=self.database_dataset,
            )
            self.inferencer.save(self.cfg.frames_path, frames=frames)

        self.inferencer.build_sequence_recall_benchmark_report(
            database_dataset=self.database_dataset, 
            similarity_kwargs=self.cfg.similarity_kwargs,
            seq_lengths=self.cfg.seq_lengths,
            per_frame_k_used=self.cfg.per_frame_k_used,
            final_k=self.cfg.final_k,
            save_dir=self.cfg.bench_report_path,
            std_mode=self.cfg.std_mode,
            scene_df_field=self.cfg.scene_df_field,
            pose_df_field=self.cfg.pose_df_field,
            recall_at_k=self.cfg.recall_at_k,
            seq_filter_kwargs=self.cfg.seq_filter_kwargs
        )

    def save(self):
        pass

    def load(self):
        pass