from __future__ import annotations

import os
from pathlib import Path
from typing import Union

import torch
from torch import Tensor, nn
from torchvision import transforms as T

from gsloc.models.edtformer.network import VPRNet

PathLike = Union[str, Path]

# Official EDTformer eval uses 322x322 + ImageNet normalization (see Tong-Jin01/EDTformer parser.py).
INPUT_SIZE = 322
DESCRIPTOR_DIM = 4096  # 256 (channel_proj) * 16 (row_proj)
EDTFORMER_WEIGHTS = Path(__file__).resolve().parents[3] / "weights" / "EDTformer.pth"


def load_edtformer_checkpoint(model: nn.Module, weights: PathLike, device: str | torch.device = "cpu") -> nn.Module:
    """Load EDTformer checkpoint (``model_state_dict`` or raw state dict)."""
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


def _default_edtformer_weights() -> Path:
    env_path = os.environ.get("GSLoc_EDTFORMER_WEIGHTS")
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GSLoc_EDTFORMER_WEIGHTS points to missing file: {path}")
        return path
    if EDTFORMER_WEIGHTS.is_file():
        return EDTFORMER_WEIGHTS
    raise FileNotFoundError(
        f"EDTformer weights not found at {EDTFORMER_WEIGHTS}. "
        "Download EDTformer.pth from https://github.com/Tong-Jin01/EDTformer "
        "or set GSLoc_EDTFORMER_WEIGHTS."
    )


def build_edtformer_net() -> VPRNet:
    return VPRNet()


def get_edtformer_image_transform(resize: tuple[int, int] = (INPUT_SIZE, INPUT_SIZE)) -> T.Compose:
    """Image preprocessing matching the official EDTformer benchmark (hard_resize + ImageNet norm)."""
    return T.Compose(
        [
            T.ToTensor(),
            T.Resize(list(resize), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def _images_from_batch(batch: dict[str, Tensor]) -> Tensor:
    key = next((k for k in batch if k.startswith("images_")), None)
    if key is None:
        raise KeyError("No key starting with 'images_' found in the batch.")
    return batch[key]


class EDTformer(nn.Module):
    """EDTformer global image descriptor for place recognition.

    Vendored from https://github.com/Tong-Jin01/EDTformer. Expects RGB images at 322x322
    with ImageNet normalization (use :func:`get_edtformer_image_transform`). Outputs a single
    L2-normalized ``final_descriptor`` of shape ``[B, 4096]``.
    """

    def __init__(
        self,
        weights: PathLike | None = None,
        device: str | torch.device = "cpu",
        model: VPRNet | None = None,
    ) -> None:
        super().__init__()
        self.model = model if model is not None else build_edtformer_net()
        if model is None:
            if weights is None:
                weights = _default_edtformer_weights()
            load_edtformer_checkpoint(self.model, weights, device=device)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        images = _images_from_batch(batch)
        return {"final_descriptor": self.model(images)}
