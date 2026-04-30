from gsloc.inference.pr_infer import PerFramePR
import numpy as np

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


def TwoModelsFrameSharedSumMerge(
    frames_a: list[PerFramePR],
    frames_b: list[PerFramePR],
    *,
    distance_coef_a: float = 1.0,
    distance_coef_b: float = 1.0,
) -> list[PerFramePR]:
    """Keep only shared indices and sum their corresponding distances.

    For each frame pair, this function intersects ``indices`` from ``frames_a`` and
    ``frames_b``. The output contains only shared indices; each output distance is
    ``dist_a + dist_b`` for the same index. Results are sorted by summed distance
    (ascending).

    Args:
        frames_a: First model's per-frame PR list.
        frames_b: Second model's per-frame PR list (same length as ``frames_a``).

    Returns:
        One ``PerFramePR`` per input frame containing only shared candidates.

    Raises:
        ValueError: If the two frame lists have different lengths.
    """
    if len(frames_a) != len(frames_b):
        raise ValueError(
            f"Frame list length mismatch: {len(frames_a)} != {len(frames_b)}"
        )

    merged: list[PerFramePR] = []
    for fa, fb in zip(frames_a, frames_b):
        idx_a = np.asarray(fa.indices, dtype=np.int64, copy=False)
        idx_b = np.asarray(fb.indices, dtype=np.int64, copy=False)
        dist_a = np.asarray(fa.distances, dtype=np.float32, copy=False)
        dist_b = np.asarray(fb.distances, dtype=np.float32, copy=False)

        b_dist_by_idx = {int(i): float(d) for i, d in zip(idx_b.tolist(), dist_b.tolist())}
        shared_mask_a = np.isin(idx_a, idx_b, assume_unique=False)
        shared_idx = idx_a[shared_mask_a]
        if shared_idx.size == 0:
            merged.append(
                PerFramePR(
                    indices=np.empty((0,), dtype=np.int64),
                    distances=np.empty((0,), dtype=np.float32),
                    db_idx=np.empty((0,), dtype=np.int64) if fa.db_idx is not None else None,
                )
            )
            continue

        shared_dist_a = dist_a[shared_mask_a]
        shared_dist = np.array(
            [distance_coef_a * da + distance_coef_b * b_dist_by_idx[int(i)] for i, da in zip(shared_idx, shared_dist_a)],
            dtype=np.float32,
        )

        order = np.argsort(shared_dist, kind="mergesort")
        out_idx = shared_idx[order].astype(np.int64, copy=False)
        out_dist = shared_dist[order].astype(np.float32, copy=False)

        if fa.db_idx is not None:
            db_a = np.asarray(fa.db_idx, dtype=np.int64, copy=False)
            out_db_idx: np.ndarray | None = db_a[shared_mask_a][order].astype(
                np.int64, copy=False
            )
        else:
            out_db_idx = None

        merged.append(PerFramePR(indices=out_idx, distances=out_dist, db_idx=out_db_idx))
    return merged