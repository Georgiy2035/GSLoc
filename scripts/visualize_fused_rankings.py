#!/usr/bin/env python3
"""Visualize query vs top-K fused PR candidates from fused_rankings.npz."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.transforms import blended_transform_factory
import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_FUSED = (
    ROOT
    / "data/tests/26-06-01/3RScan/Makarov/64xMegaloc/rerank_k_1000_per_frame_k_25"
    / "base_seq_report/room-sim/001-window/fused_rankings.npz"
)
DEFAULT_QUERY_META = ROOT / "data/tests/26-06-01/3RScan/cache/query_cache/Megaloc/meta.parquet"
DEFAULT_DB_META = ROOT / "data/tests/26-06-01/3RScan/cache/indexes/Megaloc/meta.parquet"
DEFAULT_FUSED_BOTTOM = (
    ROOT
    / "data/tests/26-06-01/3RScan/Makarov/64xMegaloc/rerank_k_1000_per_frame_k_25"
    / "base_seq_report/room-sim/035-window/fused_rankings.npz"
)
DEFAULT_FRAMES = ROOT / "data/tests/26-05-06/makarov/64_pure_graph/frames.npz"
DEFAULT_OUTPUT = ROOT / "data/plots/fused_rankings_query_candidates.png"

ROW_LABELS = {
    "frames": "Graph descriptor\nretrieval output",
    "fused": "Image descriptor\nreranking",
    "fused_bottom": "Sequence\nagregation result",
}
ROW_LABEL_FONTSIZE = 13
QUERY_LABEL = "Query image"
QUERY_SEQUENCE_LABEL = "Query sequence"
BORDER_LINEWIDTH = 5.0


def _rank_label_extra_height_ratio(rank_fontsize: int) -> float:
    """Gridspec height for the shared Top-K label strip above candidate rows."""
    return 0.08 * (rank_fontsize / 12)

# Keep labels narrow while preserving the original candidate-column width.
LABEL_COL_RATIO = 0.65
_ORIG_PANEL_LABEL_RATIO = 0.85
_ORIG_PANEL_CAND_RATIO = 2.5
CAND_COL_RATIO = _ORIG_PANEL_CAND_RATIO * LABEL_COL_RATIO / _ORIG_PANEL_LABEL_RATIO


def _crop_square_top_bottom(img: np.ndarray) -> np.ndarray:
    """Center-crop to a square by removing equal strips from top and bottom."""
    height, width = img.shape[:2]
    if height <= width:
        return img
    offset = (height - width) // 2
    return img[offset : offset + width, :]


def _load_rgb(path: str | Path) -> np.ndarray:
    img = np.asarray(Image.open(path).convert("RGB"))
    img = np.rot90(img, k=-1)
    return _crop_square_top_bottom(img)


def find_suitable_query(
    is_pos: np.ndarray,
    *,
    top_k: int,
    min_pos: int = 2,
    min_neg: int = 1,
    prefer_mixed_early: bool = True,
) -> int:
    """Pick a query with enough positives and negatives among top-K candidates."""
    top = is_pos[:, :top_k]
    pos_count = top.sum(axis=1)
    neg_count = top_k - pos_count
    mask = (pos_count >= min_pos) & (neg_count >= min_neg)
    candidates = np.flatnonzero(mask)
    if candidates.size == 0:
        raise RuntimeError(
            f"No query found with >={min_pos} positives and >={min_neg} negatives in top-{top_k}"
        )

    if prefer_mixed_early:
        # Prefer queries where a wrong candidate appears early in the ranking.
        for qid in candidates:
            flags = top[qid]
            if flags[0] and (not flags[1]) and flags[2:].sum() >= 1:
                return int(qid)

    return int(candidates[0])


def _add_image_border(ax: plt.Axes, color: str, linewidth: float = 5.0) -> None:
    xmin, xmax = ax.get_xlim()
    ymin, ymax = ax.get_ylim()
    left = min(xmin, xmax)
    right = max(xmin, xmax)
    bottom = min(ymin, ymax)
    top = max(ymin, ymax)
    ax.add_patch(
        Rectangle(
            (left, bottom),
            right - left,
            top - bottom,
            fill=False,
            edgecolor=color,
            linewidth=linewidth,
            clip_on=False,
            zorder=10,
        )
    )


def _room_sim_is_pos(query_row: pd.Series, db_row: pd.Series) -> bool:
    q_room = query_row.get("room")
    d_room = db_row.get("room")
    return q_room is not None and d_room is not None and q_room == d_room


def _load_candidate_row_from_fused(
    *,
    fused_path: Path,
    query_id: int,
    db_df: pd.DataFrame,
    top_k: int,
) -> tuple[list[np.ndarray], list[bool]]:
    data = np.load(fused_path)
    db_idx = data["db_idx"]
    is_pos = data["is_pos"]

    cand_imgs: list[np.ndarray] = []
    cand_flags: list[bool] = []
    for rank in range(top_k):
        db_row_id = int(db_idx[query_id, rank])
        db_row = db_df.iloc[db_row_id]
        cand_imgs.append(_load_rgb(db_row["image_path"]))
        cand_flags.append(bool(is_pos[query_id, rank]))
    return cand_imgs, cand_flags


def _load_candidate_row_from_frames(
    *,
    frames_path: Path,
    query_id: int,
    q_df: pd.DataFrame,
    db_df: pd.DataFrame,
    top_k: int,
) -> tuple[list[np.ndarray], list[bool]]:
    data = np.load(frames_path)
    db_idx = data["db_idx"]

    query_row = q_df.iloc[query_id]
    cand_imgs: list[np.ndarray] = []
    cand_flags: list[bool] = []
    for rank in range(top_k):
        db_row_id = int(db_idx[query_id, rank])
        db_row = db_df.iloc[db_row_id]
        cand_imgs.append(_load_rgb(db_row["image_path"]))
        cand_flags.append(_room_sim_is_pos(query_row, db_row))
    return cand_imgs, cand_flags


def _stroke_half_width_data(ax: plt.Axes, linewidth_points: float) -> tuple[float, float]:
    """Convert a matplotlib stroke width from points to half-width in data units."""
    fig = ax.figure
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bbox = ax.get_window_extent(renderer)
    half_px = (linewidth_points / 72.0 * fig.dpi) / 2.0
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    span_x = abs(x1 - x0)
    span_y = abs(y1 - y0)
    if bbox.width <= 0 or bbox.height <= 0:
        half = linewidth_points / 2.0
        return half, half
    return half_px * span_x / bbox.width, half_px * span_y / bbox.height


def _draw_rank_labels(
    fig: plt.Figure,
    parent_gs,
    *,
    top_k: int,
    tile_width: int,
    x_reference_ax: plt.Axes,
    rank_fontsize: int = 12,
) -> plt.Axes:
    ax = fig.add_subplot(parent_gs)
    ax.set_xlim(x_reference_ax.get_xlim())
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.margins(0)
    label_transform = blended_transform_factory(x_reference_ax.transData, ax.transAxes)
    for rank in range(top_k):
        ax.text(
            tile_width * (rank + 0.5),
            0.0,
            f"Top-{rank + 1}",
            transform=label_transform,
            ha="center",
            va="bottom",
            fontsize=rank_fontsize,
            color="black",
            clip_on=False,
        )
    return ax


def _draw_candidate_row(
    fig: plt.Figure,
    parent_gs,
    cand_imgs: list[np.ndarray],
    cand_flags: list[bool],
    top_k: int,
    *,
    sharex: plt.Axes | None = None,
) -> plt.Axes:
    h, w = cand_imgs[0].shape[:2]
    total_w = w * top_k
    composite = np.concatenate(cand_imgs, axis=1)

    if sharex is not None:
        ax = fig.add_subplot(parent_gs, sharex=sharex)
    else:
        ax = fig.add_subplot(parent_gs)
    ax.imshow(composite, extent=(0, total_w, h, 0), aspect="equal")
    ax.set_xlim(0, total_w)
    ax.set_ylim(h, 0)
    ax.set_anchor("W")
    ax.axis("off")
    ax.margins(0)

    half_x, half_y = _stroke_half_width_data(ax, BORDER_LINEWIDTH)
    for rank in range(top_k):
        color = "#2ecc71" if cand_flags[rank] else "#e74c3c"
        ax.add_patch(
            Rectangle(
                (w * rank + half_x, half_y),
                w - 2 * half_x,
                h - 2 * half_y,
                fill=False,
                edgecolor=color,
                linewidth=BORDER_LINEWIDTH,
                clip_on=False,
                zorder=10,
            )
        )
    return ax


def _draw_query_sequence_header(
    fig: plt.Figure,
    parent_gs,
    *,
    query_img: np.ndarray,
    prev_query_img: np.ndarray,
    top_k: int,
    label_fontsize: int = ROW_LABEL_FONTSIZE,
    dots_fontsize: int = 18,
) -> None:
    label_row_ratio = 0.07 * (label_fontsize / ROW_LABEL_FONTSIZE)
    query_inner = parent_gs.subgridspec(
        2,
        top_k,
        height_ratios=[label_row_ratio, 1.0 - label_row_ratio],
        hspace=0.02,
        wspace=0.0,
    )
    seq_start = (top_k - 3) // 2
    query_col = seq_start
    dots_col = seq_start + 1
    prev_col = seq_start + 2

    ax_q_label = fig.add_subplot(query_inner[0, query_col])
    ax_q_label.axis("off")
    ax_q_label.text(
        0.5,
        0.0,
        QUERY_LABEL,
        ha="center",
        va="bottom",
        fontsize=label_fontsize,
        color="black",
    )
    ax_q = fig.add_subplot(query_inner[1, query_col])
    ax_q.imshow(query_img)
    ax_q.axis("off")

    ax_seq_label = fig.add_subplot(query_inner[0, prev_col])
    ax_seq_label.axis("off")
    ax_seq_label.text(
        0.5,
        0.0,
        QUERY_SEQUENCE_LABEL,
        ha="center",
        va="bottom",
        fontsize=label_fontsize,
        color="black",
    )

    ax_dots = fig.add_subplot(query_inner[1, dots_col])
    ax_dots.axis("off")
    ax_dots.text(
        0.5,
        0.5,
        "• • •",
        ha="center",
        va="center",
        fontsize=dots_fontsize,
        color="black",
    )

    ax_prev = fig.add_subplot(query_inner[1, prev_col])
    ax_prev.imshow(prev_query_img)
    ax_prev.axis("off")


def visualize_fused_rankings(
    *,
    fused_path: Path,
    query_meta_path: Path,
    db_meta_path: Path,
    output_path: Path,
    frames_path: Path | None = DEFAULT_FRAMES,
    fused_bottom_path: Path | None = DEFAULT_FUSED_BOTTOM,
    query_id: int | None = None,
    top_k: int = 5,
    dpi: int = 150,
) -> int:
    primary = np.load(fused_path)
    is_pos = primary["is_pos"]

    q_df = pd.read_parquet(query_meta_path)
    db_df = pd.read_parquet(db_meta_path)

    if query_id is None:
        query_id = find_suitable_query(is_pos, top_k=top_k)
    else:
        flags = is_pos[query_id, :top_k]
        if flags.sum() < 2 or (~flags).sum() < 1:
            raise ValueError(
                f"query_id={query_id} does not satisfy the requirement "
                f"(need >=2 positives and >=1 negative in top-{top_k})"
            )

    query_row = q_df.iloc[query_id]
    query_img = _load_rgb(query_row["image_path"])
    prev_query_id = query_id - 5
    if prev_query_id < 0:
        raise ValueError(f"query_id={query_id} is too small for a 5-frame lookback")
    prev_query_img = _load_rgb(q_df.iloc[prev_query_id]["image_path"])

    candidate_rows: list[tuple[str, list[np.ndarray], list[bool]]] = []
    if frames_path is not None:
        cand_imgs, cand_flags = _load_candidate_row_from_frames(
            frames_path=frames_path,
            query_id=query_id,
            q_df=q_df,
            db_df=db_df,
            top_k=top_k,
        )
        candidate_rows.append((ROW_LABELS["frames"], cand_imgs, cand_flags))
    cand_imgs, cand_flags = _load_candidate_row_from_fused(
        fused_path=fused_path,
        query_id=query_id,
        db_df=db_df,
        top_k=top_k,
    )
    candidate_rows.append((ROW_LABELS["fused"], cand_imgs, cand_flags))
    if fused_bottom_path is not None:
        cand_imgs, cand_flags = _load_candidate_row_from_fused(
            fused_path=fused_bottom_path,
            query_id=query_id,
            db_df=db_df,
            top_k=top_k,
        )
        candidate_rows.append((ROW_LABELS["fused_bottom"], cand_imgs, cand_flags))

    n_rows = len(candidate_rows)
    rank_extra = _rank_label_extra_height_ratio(ROW_LABEL_FONTSIZE)
    fig = plt.figure(figsize=(15.5, 2.5 * (n_rows + 1 + rank_extra)), facecolor="white")
    gs = fig.add_gridspec(
        2 + n_rows,
        2,
        width_ratios=[LABEL_COL_RATIO, CAND_COL_RATIO],
        height_ratios=[1.0, rank_extra] + [1.0] * n_rows,
        wspace=0.0,
        hspace=0.18,
    )

    _draw_query_sequence_header(
        fig,
        gs[0, 1],
        query_img=query_img,
        prev_query_img=prev_query_img,
        top_k=top_k,
    )

    tile_width = candidate_rows[0][1][0].shape[1]
    first_label, first_imgs, first_flags = candidate_rows[0]
    ax_label = fig.add_subplot(gs[2, 0])
    ax_label.axis("off")
    ax_label.text(
        1.0,
        0.5,
        first_label,
        ha="right",
        va="center",
        fontsize=ROW_LABEL_FONTSIZE,
        linespacing=1.25,
    )
    ref_ax = _draw_candidate_row(
        fig,
        gs[2, 1],
        first_imgs,
        first_flags,
        top_k,
    )
    _draw_rank_labels(
        fig,
        gs[1, 1],
        top_k=top_k,
        tile_width=tile_width,
        x_reference_ax=ref_ax,
    )

    for row_idx, (row_label, cand_imgs, cand_flags) in enumerate(candidate_rows[1:], start=1):
        ax_label = fig.add_subplot(gs[row_idx + 2, 0])
        ax_label.axis("off")
        ax_label.text(
            1.0,
            0.5,
            row_label,
            ha="right",
            va="center",
            fontsize=ROW_LABEL_FONTSIZE,
            linespacing=1.25,
        )
        _draw_candidate_row(
            fig,
            gs[row_idx + 2, 1],
            cand_imgs,
            cand_flags,
            top_k,
            sharex=ref_ax,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Selected query_id={query_id}")
    print(f"Saved visualization to {output_path}")
    return query_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--fused", type=Path, default=DEFAULT_FUSED)
    p.add_argument(
        "--frames",
        type=Path,
        default=DEFAULT_FRAMES,
        help="Optional frames.npz for the top candidate row (use 'none' to disable)",
    )
    p.add_argument(
        "--fused-bottom",
        type=Path,
        default=DEFAULT_FUSED_BOTTOM,
        help="Optional second fused_rankings.npz for a bottom candidate row (use 'none' to disable)",
    )
    p.add_argument("--query-meta", type=Path, default=DEFAULT_QUERY_META)
    p.add_argument("--db-meta", type=Path, default=DEFAULT_DB_META)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--query-id", type=int, default=None)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    frames = None if str(args.frames).lower() == "none" else args.frames.resolve()
    fused_bottom = None if str(args.fused_bottom).lower() == "none" else args.fused_bottom.resolve()
    visualize_fused_rankings(
        fused_path=args.fused.resolve(),
        query_meta_path=args.query_meta.resolve(),
        db_meta_path=args.db_meta.resolve(),
        output_path=args.output.resolve(),
        frames_path=frames,
        fused_bottom_path=fused_bottom,
        query_id=args.query_id,
        top_k=args.top_k,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
