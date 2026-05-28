from __future__ import annotations

import http.cookiejar
import os
import re
from pathlib import Path
from typing import Union
from urllib.request import HTTPCookieProcessor, Request, build_opener

import torch
from torch import Tensor, nn

from gsloc.models.fol.network import FoLNet

PathLike = Union[str, Path]

# Official FoL_base.pth sources (see https://github.com/chenshunpeng/FoL)
FOL_BASE_GDRIVE_FILE_ID = "1Z05ZLFliQXOPJMH1YPdXqYjzC15-0nam"
FOL_BASE_HF_REPO = "shunpeng/FoL"
FOL_BASE_HF_FILENAME = "FoL_base.pth"
FOL_BASE_CACHE_DIR = Path.home() / ".cache" / "gsloc" / "weights"
FOL_BASE_CACHE_PATH = FOL_BASE_CACHE_DIR / FOL_BASE_HF_FILENAME
# <repo_root>/weights/FoL_base.pth (repo_root = src/gsloc/models -> parents[3])
FOL_BASE_PROJECT_PATH = Path(__file__).resolve().parents[3] / "weights" / FOL_BASE_HF_FILENAME


def load_fol_checkpoint(model: nn.Module, weights: PathLike, device: str | torch.device = "cpu") -> nn.Module:
    """Load FoL checkpoint (raw state dict or nested ``model_state_dict``)."""
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


def _download_google_drive_file(file_id: str, destination: Path) -> None:
    """Download a public Google Drive file (stdlib only, with large-file confirm cookies)."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    base_url = "https://drive.google.com/uc?export=download"
    cookie_jar = http.cookiejar.CookieJar()
    opener = build_opener(HTTPCookieProcessor(cookie_jar))

    def _fetch(url: str, timeout: int = 300) -> bytes:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with opener.open(req, timeout=timeout) as resp:
            return resp.read()

    data = _fetch(f"{base_url}&id={file_id}", timeout=120)

    confirm_token = None
    for cookie in cookie_jar:
        if cookie.name.startswith("download_warning"):
            confirm_token = cookie.value
            break
    if confirm_token is None:
        token_match = re.search(r"confirm=([0-9A-Za-z_]+)", data.decode("utf-8", errors="ignore"))
        if token_match:
            confirm_token = token_match.group(1)

    if confirm_token:
        data = _fetch(f"{base_url}&id={file_id}&confirm={confirm_token}", timeout=600)

    if data[:15].lower().startswith(b"<!doctype") or data[:6].lower().startswith(b"<html"):
        raise RuntimeError(
            f"Google Drive returned HTML instead of a checkpoint for id={file_id}. "
            "Download FoL_base.pth manually and pass weights=... to FoLBase."
        )
    destination.write_bytes(data)


def _download_hf_fol_base(destination: Path) -> None:
    from huggingface_hub import hf_hub_download

    downloaded = Path(hf_hub_download(repo_id=FOL_BASE_HF_REPO, filename=FOL_BASE_HF_FILENAME))
    destination.parent.mkdir(parents=True, exist_ok=True)
    if downloaded.resolve() != destination.resolve():
        destination.write_bytes(downloaded.read_bytes())


def _default_fol_base_weights() -> Path:
    env_path = os.environ.get("GSLoc_FOL_BASE_WEIGHTS")
    if env_path:
        path = Path(env_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"GSLoc_FOL_BASE_WEIGHTS points to missing file: {path}")
        return path

    if FOL_BASE_PROJECT_PATH.is_file():
        return FOL_BASE_PROJECT_PATH

    if FOL_BASE_CACHE_PATH.is_file():
        return FOL_BASE_CACHE_PATH

    errors: list[str] = []

    # Hugging Face is usually more reliable than Drive for automated downloads.
    try:
        _download_hf_fol_base(FOL_BASE_CACHE_PATH)
        return FOL_BASE_CACHE_PATH
    except ImportError:
        errors.append("Hugging Face: huggingface_hub is not installed (pip install huggingface_hub)")
    except Exception as exc:
        errors.append(f"Hugging Face: {exc}")

    try:
        _download_google_drive_file(FOL_BASE_GDRIVE_FILE_ID, FOL_BASE_CACHE_PATH)
        return FOL_BASE_CACHE_PATH
    except Exception as exc:
        errors.append(f"Google Drive: {exc}")

    raise RuntimeError(
        "Failed to download FoL_base.pth. Tried Google Drive and Hugging Face.\n"
        + "\n".join(f"  - {e}" for e in errors)
        + f"\nManual download: https://drive.google.com/file/d/{FOL_BASE_GDRIVE_FILE_ID}/view"
        + f"\nThen: FoLBase(weights='/path/to/FoL_base.pth') "
        + f"or export GSLoc_FOL_BASE_WEIGHTS=/path/to/FoL_base.pth"
    )


def build_fol_base_net() -> FoLNet:
    return FoLNet(
        num_channels=768,
        model_name="dinov2_vitb14",
        num_trainable_blocks=4,
        backbone_pretrained=False,
    )


def _images_from_batch(batch: dict[str, Tensor]) -> Tensor:
    key = next((k for k in batch if k.startswith("images_")), None)
    if key is None:
        raise KeyError("No key starting with 'images_' found in the batch.")
    return batch[key]


class FoLBase(nn.Module):
    """FoL ViT-B global descriptor (FoL-global stage)."""

    def __init__(
        self,
        weights: PathLike | None = None,
        device: str | torch.device = "cpu",
        model: FoLNet | None = None,
    ) -> None:
        super().__init__()
        self.model = model if model is not None else build_fol_base_net()
        if model is None:
            if weights is None:
                weights = _default_fol_base_weights()
            load_fol_checkpoint(self.model, weights, device=device)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        global_descriptor, *_ = self.model(_images_from_batch(batch), test=True)
        return {"final_descriptor": global_descriptor}


class FoLBaseRerank(nn.Module):
    """FoL ViT-B local features as a flat descriptor for ``PlaceRecognitionRerankPipeline``.

    Outputs padded discriminative local tokens (same tensor as FoL ``local_feature`` in eval),
    flattened to ``[B, L * 128]``. Use with input size 322x322 -> L=3600, D=460800.
    """

    LOCAL_FEATURE_DIM = 128

    def __init__(
        self,
        weights: PathLike | None = None,
        device: str | torch.device = "cpu",
        model: FoLNet | None = None,
    ) -> None:
        super().__init__()
        self.model = model if model is not None else build_fol_base_net()
        if model is None:
            if weights is None:
                weights = _default_fol_base_weights()
            load_fol_checkpoint(self.model, weights, device=device)

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        _global_descriptor, local_feature, *_ = self.model(_images_from_batch(batch), test=True)
        return {"final_descriptor": local_feature.flatten(1).contiguous()}
