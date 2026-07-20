from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import torch
from torch import Tensor, nn

PathLike = Union[str, Path]

SELAVPR_HUB_REPO = "Lu-Feng/SelaVPRplusplus"
SELAVPR_HUB_ENTRY = "SelaVPRplusplus"
SELAVPR_BASE_RERANK_WEIGHTS = (
    Path(__file__).resolve().parents[3] / "weights" / "SelaVPRplusplus_base_rerank.pth"
)

# Official SelaVPR++ uses 518x518 inputs (DINOv2 ViT-B/L, patch size 14).
INPUT_SIZE = 518
BINARY_DESCRIPTOR_DIM = 512
FLOAT_RERANK_DESCRIPTOR_DIM = 2048


@dataclass(frozen=True)
class SelaVPRplusplusConfig:
    backbone: str = "dinov2-base"
    aggregation: str = "gem"
    hashing: bool = True
    rerank: bool = True


def _hub_repo_dir() -> Path:
    return Path(torch.hub.get_dir()) / "Lu-Feng_SelaVPRplusplus_main"


def _ensure_hub_source() -> Path:
    """Download SelaVPR++ source into the torch hub cache if needed."""
    repo_dir = _hub_repo_dir()
    if repo_dir.is_dir():
        return repo_dir
    torch.hub.load(
        SELAVPR_HUB_REPO,
        SELAVPR_HUB_ENTRY,
        backbone="dinov2-base",
        aggregation="gem",
        hashing=True,
        rerank=True,
        trust_repo=True,
    )
    if not repo_dir.is_dir():
        raise RuntimeError(
            f"Failed to fetch SelaVPR++ sources into {repo_dir}. "
            "Check network access or clone https://github.com/Lu-Feng/SelaVPRplusplus manually."
        )
    return repo_dir


def _import_geo_localization_net():
    repo_dir = _ensure_hub_source()
    repo_str = str(repo_dir)
    if repo_str not in sys.path:
        sys.path.insert(0, repo_str)
    from model.network import GeoLocalizationNet  # type: ignore[import-not-found]

    return GeoLocalizationNet


def load_selavpr_checkpoint(model: nn.Module, weights: PathLike, device: str | torch.device = "cpu") -> nn.Module:
    """Load SelaVPR++ checkpoint (``model_state_dict`` or raw state dict)."""
    try:
        checkpoint = torch.load(weights, map_location=device, weights_only=False)
    except TypeError:
        checkpoint = torch.load(weights, map_location=device)
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif isinstance(checkpoint, dict):
        state_dict = checkpoint
    else:
        raise TypeError(f"Unexpected checkpoint type: {type(checkpoint)}")
    state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict, strict=False)
    return model


def build_selavprplusplus_base_rerank_net(config: SelaVPRplusplusConfig | None = None) -> nn.Module:
    """Build GeoLocalizationNet (dinov2-base, GeM, hashing + rerank) without pretrained hub weights."""
    cfg = config or SelaVPRplusplusConfig()
    if not (cfg.hashing and cfg.rerank):
        raise ValueError("build_selavprplusplus_base_rerank_net expects hashing=True and rerank=True")
    GeoLocalizationNet = _import_geo_localization_net()

    class _Args:
        backbone = cfg.backbone
        aggregation = cfg.aggregation
        hashing = cfg.hashing
        rerank = cfg.rerank
        resume = True

    return GeoLocalizationNet(_Args())


def _default_selavpr_base_rerank_weights() -> Path:
    env_path = os.environ.get("GSLoc_SELAVPR_BASE_RERANK_WEIGHTS")
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GSLoc_SELAVPR_BASE_RERANK_WEIGHTS points to missing file: {path}")
        return path
    if SELAVPR_BASE_RERANK_WEIGHTS.is_file():
        return SELAVPR_BASE_RERANK_WEIGHTS
    raise FileNotFoundError(
        f"SelaVPR++ rerank weights not found at {SELAVPR_BASE_RERANK_WEIGHTS}. "
        "Download SelaVPRplusplus_base_rerank.pth or set GSLoc_SELAVPR_BASE_RERANK_WEIGHTS."
    )


def _images_from_batch(batch: dict[str, Tensor]) -> Tensor:
    key = next((k for k in batch if k.startswith("images_")), None)
    if key is None:
        raise KeyError("No key starting with 'images_' found in the batch.")
    return batch[key]


class SelaVPRplusplusBaseRerank(nn.Module):
    """SelaVPR++ ViT-B with binary retrieval + float rerank descriptors.

    Matches the official two-branch model (``hashing=True``, ``rerank=True``) from
    https://github.com/Lu-Feng/SelaVPRplusplus. Expects RGB images at 518x518 (ImageNet norm
    as in the VPR benchmark). Outputs:

    - ``final_descriptor``: STE-binarized branch, shape ``[B, 512]`` with values in ``{-1, +1}``
    - ``rerank_descriptor``: L2-normalized float branch, shape ``[B, 2048]``

    Use with ``PlaceRecognitionRerankPipeline`` (binary index on ``final_descriptor``,
    float rerank on ``rerank_descriptor``) or build separate indices as in ``eval_rerank.py``.
    """

    def __init__(
        self,
        weights: PathLike | None = None,
        device: str | torch.device = "cpu",
        model: nn.Module | None = None,
        config: SelaVPRplusplusConfig | None = None,
    ) -> None:
        super().__init__()
        self.config = config or SelaVPRplusplusConfig()
        self.model = model if model is not None else build_selavprplusplus_base_rerank_net(self.config)
        if model is None:
            if weights is None:
                weights = _default_selavpr_base_rerank_weights()
            load_selavpr_checkpoint(self.model, weights, device=device)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        images = _images_from_batch(batch)
        _z_float, z_binary, x_rerank = self.model(images)
        return {
            "final_descriptor": z_binary,
            "rerank_descriptor": x_rerank,
        }


class SelaVPRplusplusBaseRerankFloat(nn.Module):
    """Same backbone as :class:`SelaVPRplusplusBaseRerank`, but only the float rerank branch.

    Useful when another model handles coarse retrieval and this head only supplies
    ``final_descriptor`` for L2/FAISS reranking (2048-d, L2-normalized).
    """

    def __init__(
        self,
        weights: PathLike | None = None,
        device: str | torch.device = "cpu",
        model: nn.Module | None = None,
        config: SelaVPRplusplusConfig | None = None,
    ) -> None:
        super().__init__()
        self._wrapper = SelaVPRplusplusBaseRerank(
            weights=weights,
            device=device,
            model=model,
            config=config,
        )
        self.model = self._wrapper.model

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        out = self._wrapper(batch)
        return {"final_descriptor": out["rerank_descriptor"]}
