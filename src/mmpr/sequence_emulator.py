from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from opr.inference.pipelines.sequence_place_recognition import _candidate_pool_fusion as _cpf

from mmpr.pr_cache import PerFramePR


@dataclass
class EmulatorConfig:
    max_window: int = 20
    per_frame_k_used: int | None = None  # if set, truncate per-frame results to this K before fusion
    final_k: int = 10
    recency_weighting: Literal["none", "linear", "exp"] = "none"


def emulate_sequence_fusion(
    frames: list[PerFramePR],
    cfg: EmulatorConfig,
    scene_data: list[str] | None = None,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Emulate SequencePlaceRecognitionPipeline fusion over cached per-frame PR results.

    Returns per-query fused (distances, indices). Different final_k can be obtained by slicing.
    """
    max_window = int(cfg.max_window)
    per_k_used = int(cfg.per_frame_k_used) if cfg.per_frame_k_used is not None else None
    final_k = int(cfg.final_k)

    fused: list[tuple[np.ndarray, np.ndarray]] = []

    # Rolling window buffers
    win_d: list[np.ndarray] = []
    win_i: list[np.ndarray] = []

    for i, f in enumerate(frames):
        di = f.distances if per_k_used is None else f.distances[:per_k_used]
        ii = f.indices if per_k_used is None else f.indices[:per_k_used]
        
        win_d.append(di.astype(np.float32, copy=False))
        win_i.append(ii.astype(np.int64, copy=False))

        if scene_data is not None: 
            if scene_data[i] != scene_data[i-1]:
                while len(win_d) > 1:
                    win_d.pop(0)
                    win_i.pop(0)
        
        if len(win_d) > max_window:
            win_d.pop(0)
            win_i.pop(0)

        if win_d:
            per_d = np.stack(win_d, axis=0)
            per_i = np.stack(win_i, axis=0)
        else:
            per_d = np.empty((0, 0), dtype=np.float32)
            per_i = np.empty((0, 0), dtype=np.int64)

        # Optional recency weighting
        if cfg.recency_weighting != "none" and per_d.size > 0:
            N = per_d.shape[0]
            ages = np.arange(N - 1, -1, -1, dtype=np.float32)
            if cfg.recency_weighting == "linear":
                weights = 1.0 + ages / max(1, N - 1)
            else:
                base = 1.25
                weights = base ** (ages / max(1, N - 1))
            per_d = per_d * weights[:, None]

        fused_d, fused_i = _cpf(per_d, per_i, final_k)
        fused.append((fused_d, fused_i))

    return fused


