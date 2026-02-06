from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal

import numpy as np
import torch
from torch import Tensor

from opr.inference.index import FaissFlatIndex
from opr.inference.pipelines import PlaceRecognitionPipeline
from opr.inference.preprocessing import PointCloudMinkPreprocessor
from opr.models.place_recognition import MinkLoc3D
from opr.models.place_recognition.base import LateFusionModel
from opr.utils import parse_device

from mmpr.data.transforms import get_T_map_to_world
from mmpr.data.pcd import SimplePCDLoader
from mmpr.data.image import get_default_image_transform, SimpleImageLoader
from mmpr.pr_cache import PerFramePR, save_pr_cache_npz
from mmpr.models import MegaLoc
from mmpr.data.multimodal import SimpleMultimodalLoader


@dataclass
class PRInferConfig:
    root_data_dir: Path
    map_name: str
    db_map_dir: Path
    index_dir: Path
    weights: Optional[Path] = None
    device: str = "cuda"
    per_frame_k: int = 100
    pr_quant_size: float = 0.05


class ImagePreprocessor:
    def __call__(self, image: Tensor) -> dict[str, Tensor]:
        return {"images_0": image.unsqueeze(0)}


class MultimodalPreprocessor:
    
    def __init__(self, pr_quant_size: float = 0.05, use_intensity: bool = False) -> None:
        self.pc_pre = PointCloudMinkPreprocessor(quantization_size=pr_quant_size, use_intensity=use_intensity)
        self.image_pre = ImagePreprocessor()
    
    def __call__(self, points: np.ndarray, image: Tensor) -> dict[str, Tensor]:
        image_input = self.image_pre(image)
        pc_input = self.pc_pre(points)
        return {**image_input, **pc_input}


class PRInferencer:
    """Run PlaceRecognitionPipeline over a map and cache per-frame PR results."""

    def __init__(self, cfg: PRInferConfig, model: Literal["minkloc3d", "megaloc", "mssplace"] = "minkloc3d") -> None:
        self.cfg = cfg
        self.device = parse_device(cfg.device)
        self.index = FaissFlatIndex.load(str(cfg.index_dir))
        if model == "minkloc3d":
            if cfg.weights is None:
                raise ValueError("weights must be provided for minkloc3d")
            self.model = MinkLoc3D()
        elif model == "megaloc":
            if cfg.weights is not None:
                raise ValueError("weights must not be provided for megaloc")
            self.model = MegaLoc()
        elif model == "mssplace":
            if cfg.weights is None:
                raise ValueError("weights must be provided for mssplace")
            lidar_model = MinkLoc3D()
            lidar_model.load_state_dict(torch.load(cfg.weights), strict=True)
            rgb_model = MegaLoc()
            self.model = LateFusionModel(image_module=rgb_model, cloud_module=lidar_model)
        else:
            raise ValueError(f"Invalid model: {model}")
        self.pr = PlaceRecognitionPipeline(
            index=self.index,
            model=self.model,
            model_weights_path=cfg.weights if model != "mssplace" else None,
            device=self.device,
        )
        if model == "minkloc3d":
            self.pre = PointCloudMinkPreprocessor(quantization_size=float(cfg.pr_quant_size), use_intensity=False)
        elif model == "megaloc":
            self.pre = ImagePreprocessor()
        elif model == "mssplace":
            self.pre = MultimodalPreprocessor(pr_quant_size=float(cfg.pr_quant_size), use_intensity=False)

        map_dir = (cfg.root_data_dir / cfg.map_name / "keyframe_map").resolve()
        T_map_to_world = get_T_map_to_world(cfg.map_name)
        if model == "minkloc3d":
            self.loader = SimplePCDLoader(map_dir, T_map_to_world=T_map_to_world)
        elif model == "megaloc":
            self.loader = SimpleImageLoader(map_dir, T_map_to_world=T_map_to_world)
        elif model == "mssplace":
            self.loader = SimpleMultimodalLoader(map_dir, T_map_to_world=T_map_to_world)

    def run(self) -> list[PerFramePR]:
        frames: list[PerFramePR] = []
        for idx in range(len(self.loader)):
            if isinstance(self.model, LateFusionModel):
                image, points, _pose7, _image_path, _scan_path = self.loader[idx]
                pr_input: dict[str, Tensor] = self.pre(points, image)
            else:
                data, _pose7, _path = self.loader[idx]
                pr_input: dict[str, Tensor] = self.pre(data)
            pr_input = {k: v.to(self.device) for k, v in pr_input.items()}
            res = self.pr.infer(pr_input, k=int(self.cfg.per_frame_k))
            frames.append(PerFramePR(indices=res.indices, distances=res.distances, db_idx=res.db_idx))
        return frames

    def save(self, output: Path, frames: Optional[list[PerFramePR]] = None) -> None:
        if frames is None:
            frames = self.run()
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        save_pr_cache_npz(output, frames)


