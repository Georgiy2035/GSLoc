from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol

import numpy as np


class MatchOracle(Protocol):
    def __call__(self, query_id: int, db_idx: int) -> bool:
        ...


def recall_at_k_per_query(db_indices_ranked: np.ndarray, is_pos: np.ndarray, ks: Iterable[int]) -> dict[int, float]:
    """Compute per-query Recall@K for a set of K.

    Args:
        db_indices_ranked: [R] ranked db indices for a query (ignored except for length)
        is_pos: [R] boolean labels for each rank (True if match)
        ks: iterable of K values
    Returns:
        dict K -> recall (0.0 or 1.0 since single-query Recall@K)
    """
    R = int(is_pos.shape[0])
    out: dict[int, float] = {}
    for k in ks:
        kk = min(int(k), R)
        out[int(k)] = 1.0 if np.any(is_pos[:kk]) else 0.0
    return out


def micro_precision_recall_from_scores(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute micro-averaged precision-recall curve from binary labels and continuous scores.

    We sort by descending score, then sweep thresholds at each unique score.
    Returns (precision, recall, thresholds) where thresholds correspond to sorted unique scores.
    """
    y_true = y_true.astype(np.bool_, copy=False)
    order = np.argsort(-scores)
    y_sorted = y_true[order]
    s_sorted = scores[order]

    # Cumulative counts of TP and FP as we include more predictions
    tp = np.cumsum(y_sorted, dtype=np.int64)
    fp = np.cumsum(~y_sorted, dtype=np.int64)
    total_pos = int(np.sum(y_true))
    # Handle degenerate case
    if total_pos == 0:
        return np.array([1.0], dtype=np.float64), np.array([0.0], dtype=np.float64), np.array([], dtype=np.float64)

    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / total_pos

    # Collapse to unique thresholds (score changes)
    uniq_thr, first_idx = np.unique(s_sorted, return_index=True)
    # Keep final index too (full list)
    keep = np.concatenate([first_idx, np.array([len(s_sorted) - 1], dtype=np.int64)])
    keep = np.unique(keep)

    return precision[keep], recall[keep], uniq_thr


def auc_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """Area under PR via trapezoidal integration (recall sorted increasing)."""
    # Ensure monotonic recall sort
    order = np.argsort(recall)
    r = recall[order]
    p = precision[order]
    return float(np.trapz(p, r))


def max_recall_at_prec1(precision: np.ndarray, recall: np.ndarray, eps: float = 1e-9) -> float:
    """Return max recall among points with precision >= 1 - eps."""
    mask = precision >= (1.0 - float(eps))
    if not np.any(mask):
        return 0.0
    return float(np.max(recall[mask]))


def f1_from_pr(precision: np.ndarray, recall: np.ndarray) -> np.ndarray:
    p = precision.astype(np.float64)
    r = recall.astype(np.float64)
    denom = np.maximum(p + r, 1e-12)
    return 2.0 * (p * r) / denom


@dataclass
class RetrievalCurves:
    precision: np.ndarray
    recall: np.ndarray
    thresholds: np.ndarray
    auc_pr: float
    f1: np.ndarray
    f1_max: float
    max_recall_at_prec1: float


def build_micro_curves_from_rankings(
    rankings: list[tuple[np.ndarray, np.ndarray]],
    oracle: MatchOracle,
    query_ids: np.ndarray | None = None,
    score_from_distance: bool = True,
) -> RetrievalCurves:
    """Build micro-averaged PR and F1 curves by pooling candidates across queries.

    Args:
        rankings: list of (distances[R], db_idx[R]) for each query (use fused or single-frame)
        oracle: callable returning True if (query_id, db_idx) is a match
        query_ids: optional per-query ids; defaults to range(len(rankings))
        score_from_distance: if True, use score = -distance
    """
    N = len(rankings)
    if query_ids is None:
        query_ids = np.arange(N, dtype=np.int64)

    all_scores: list[float] = []
    all_labels: list[bool] = []
    for qi, (dists, db_ids) in zip(query_ids, rankings):
        if dists is None or db_ids is None:
            continue
        if score_from_distance:
            sc = -np.asarray(dists, dtype=np.float64)
        else:
            sc = np.asarray(dists, dtype=np.float64)
        lbl = np.array([bool(oracle(int(qi), int(db))) for db in np.asarray(db_ids, dtype=np.int64)], dtype=np.bool_)
        all_scores.append(sc)
        all_labels.append(lbl)

    if not all_scores:
        return RetrievalCurves(
            precision=np.array([1.0], dtype=np.float64),
            recall=np.array([0.0], dtype=np.float64),
            thresholds=np.array([], dtype=np.float64),
            auc_pr=0.0,
            f1=np.array([0.0], dtype=np.float64),
            f1_max=0.0,
            max_recall_at_prec1=0.0,
        )

    scores = np.concatenate(all_scores, axis=0)
    labels = np.concatenate(all_labels, axis=0)
    P, R, T = micro_precision_recall_from_scores(labels, scores)
    A = auc_pr(P, R)
    F = f1_from_pr(P, R)
    return RetrievalCurves(precision=P, recall=R, thresholds=T, auc_pr=A, f1=F, f1_max=float(np.max(F)), max_recall_at_prec1=max_recall_at_prec1(P, R))


