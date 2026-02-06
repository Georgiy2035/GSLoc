from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


@dataclass
class PerFramePR:
    """Cached per-frame PR search result (no model/FAISS rerun needed).

    Attributes:
        indices: np.ndarray[int64] of shape [K]
        distances: np.ndarray[float32] of shape [K]
        db_idx: optional np.ndarray[int64] of shape [K]
    """

    indices: np.ndarray
    distances: np.ndarray
    db_idx: np.ndarray | None = None


def save_pr_cache_npz(path: str | Path, frames: Iterable[PerFramePR]) -> None:
    """Save a sequence of per-frame PR caches to npz.

    Stored as arrays stacked along axis 0: indices[N,K], distances[N,K], optional db_idx[N,K].
    """
    path = Path(path)
    frames_list = list(frames)
    if not frames_list:
        raise ValueError("No frames to save")

    inds = np.stack([f.indices.astype(np.int64, copy=False) for f in frames_list], axis=0)
    dists = np.stack([f.distances.astype(np.float32, copy=False) for f in frames_list], axis=0)
    has_db = all((f.db_idx is not None) for f in frames_list)
    data: dict[str, np.ndarray] = {"indices": inds, "distances": dists}
    if has_db:
        db = np.stack([f.db_idx.astype(np.int64, copy=False) for f in frames_list], axis=0)  # type: ignore[union-attr]
        data["db_idx"] = db
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(path), **data)


def load_pr_cache_npz(path: str | Path) -> list[PerFramePR]:
    """Load a sequence of cached per-frame PR results from npz."""
    path = Path(path)
    with np.load(str(path)) as f:
        inds = f["indices"].astype(np.int64, copy=False)
        dists = f["distances"].astype(np.float32, copy=False)
        if "db_idx" in f:
            db = f["db_idx"].astype(np.int64, copy=False)
        else:
            db = None
    N = inds.shape[0]
    frames: list[PerFramePR] = []
    for i in range(N):
        frames.append(PerFramePR(indices=inds[i], distances=dists[i], db_idx=(None if db is None else db[i])))
    return frames


