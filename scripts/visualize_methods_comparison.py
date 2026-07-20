#!/usr/bin/env python3
"""Compare per-method top-K PR candidates for one or two queries side by side."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from visualize_fused_rankings import (
    ROW_LABEL_FONTSIZE,
    _draw_candidate_row,
    _draw_query_sequence_header,
    _draw_rank_labels,
    _load_candidate_row_from_fused,
    _load_rgb,
    _rank_label_extra_height_ratio,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "data/tests/26-06-01/3RScan"

QUERY_META = DATASET / "cache/query_cache/Megaloc/meta.parquet"
DB_META = DATASET / "cache/indexes/Megaloc/meta.parquet"

METHOD_ROWS: list[tuple[str, Path]] = [
    ("MegaLoc", DATASET / "Megaloc/rerank_k_500_per_frame_k_25/base_seq_report/room-sim/001-window/fused_rankings.npz"),
    ("EDTFormer", DATASET / "EDTformer/rerank_k_500_per_frame_k_25/base_seq_report/room-sim/001-window/fused_rankings.npz"),
    ("FoL", DATASET / "FoL_base/rerank_k_50_per_frame_k_25/base_seq_report/room-sim/001-window/fused_rankings.npz"),
    ("SelaVPR++", DATASET / "SelaVPRpp/rerank_k_100_per_frame_k_25/base_seq_report/room-sim/001-window/fused_rankings.npz"),
    ("GraphSeqLoc", DATASET / "Makarov/64xMegaloc/rerank_k_1000_per_frame_k_25/base_seq_report/room-sim/035-window/fused_rankings.npz"),
]

DEFAULT_OUTPUT = ROOT / "data/plots/methods_comparison_query_candidates.png"
MEGALOC_ROW = 0
GRAPHSEQLOC_ROW = -1

LABEL_COL_RATIO = 0.85
CAND_COL_RATIO = 2.5
LEFT_PANEL_RATIO = LABEL_COL_RATIO + CAND_COL_RATIO
SINGLE_PANEL_FIG_W = 15.5
FONT_SCALE = 2.0
METHOD_ROW_LABEL_FONTSIZE = int(ROW_LABEL_FONTSIZE * FONT_SCALE)
METHOD_RANK_FONTSIZE = int(12 * FONT_SCALE)
METHOD_DOTS_FONTSIZE = int(18 * FONT_SCALE)


def _method_is_pos(method_rows: list[tuple[str, Path]], top_k: int) -> list[np.ndarray]:
    return [np.load(path)["is_pos"][:, :top_k] for _, path in method_rows]


def find_primary_query(method_rows: list[tuple[str, Path]], *, top_k: int) -> int:
    """MegaLoc has >=1 error; GraphSeqLoc is all-correct in top-K."""
    is_pos = _method_is_pos(method_rows, top_k)
    all_correct = is_pos[GRAPHSEQLOC_ROW].all(axis=1)
    has_error = (~is_pos[MEGALOC_ROW]).any(axis=1)
    candidates = np.flatnonzero(all_correct & has_error)
    if candidates.size == 0:
        raise RuntimeError("No primary query satisfies the row constraints")
    return int(candidates[0])


def find_secondary_query(method_rows: list[tuple[str, Path]], *, top_k: int) -> int:
    """MegaLoc has >=2 errors; GraphSeqLoc has <=1 error in top-K."""
    is_pos = _method_is_pos(method_rows, top_k)
    mega_err = (~is_pos[MEGALOC_ROW]).sum(axis=1)
    gsl_err = (~is_pos[GRAPHSEQLOC_ROW]).sum(axis=1)
    candidates = np.flatnonzero((mega_err >= 2) & (gsl_err <= 1))
    if candidates.size == 0:
        raise RuntimeError("No secondary query satisfies the row constraints")

    # Prefer a clear example: exactly 2 MegaLoc errors and 1 GraphSeqLoc error.
    for q in candidates:
        if mega_err[q] == 2 and gsl_err[q] == 1:
            return int(q)
    return int(candidates[0])


def _load_panel_rows(
    *,
    query_id: int,
    method_rows: list[tuple[str, Path]],
    db_df: pd.DataFrame,
    top_k: int,
) -> list[tuple[str, list[np.ndarray], list[bool]]]:
    rows: list[tuple[str, list[np.ndarray], list[bool]]] = []
    for label, fused_path in method_rows:
        cand_imgs, cand_flags = _load_candidate_row_from_fused(
            fused_path=fused_path,
            query_id=query_id,
            db_df=db_df,
            top_k=top_k,
        )
        rows.append((label, cand_imgs, cand_flags))
    return rows


def _draw_methods_panel(
    fig: plt.Figure,
    panel_gs,
    *,
    query_id: int,
    q_df: pd.DataFrame,
    candidate_rows: list[tuple[str, list[np.ndarray], list[bool]]],
    top_k: int,
    show_method_labels: bool = True,
    row_label_fontsize: int = METHOD_ROW_LABEL_FONTSIZE,
    rank_fontsize: int = METHOD_RANK_FONTSIZE,
    query_label_fontsize: int = METHOD_ROW_LABEL_FONTSIZE,
    dots_fontsize: int = METHOD_DOTS_FONTSIZE,
) -> None:
    prev_query_id = query_id - 5
    if prev_query_id < 0:
        raise ValueError(f"query_id={query_id} is too small for a 5-frame lookback")

    query_img = _load_rgb(q_df.iloc[query_id]["image_path"])
    prev_query_img = _load_rgb(q_df.iloc[prev_query_id]["image_path"])

    n_rows = len(candidate_rows)
    rank_extra = _rank_label_extra_height_ratio(rank_fontsize)
    panel_row_heights = [1.0, rank_extra] + [1.0] * n_rows
    if show_method_labels:
        gs = panel_gs.subgridspec(
            2 + n_rows,
            2,
            width_ratios=[LABEL_COL_RATIO, CAND_COL_RATIO],
            height_ratios=panel_row_heights,
            wspace=0.05,
            hspace=0.18,
        )
        candidates_col = 1
    else:
        gs = panel_gs.subgridspec(
            2 + n_rows,
            1,
            height_ratios=panel_row_heights,
            hspace=0.18,
        )
        candidates_col = 0

    _draw_query_sequence_header(
        fig,
        gs[0, candidates_col],
        query_img=query_img,
        prev_query_img=prev_query_img,
        top_k=top_k,
        label_fontsize=query_label_fontsize,
        dots_fontsize=dots_fontsize,
    )
    tile_width = candidate_rows[0][1][0].shape[1]
    first_label, first_imgs, first_flags = candidate_rows[0]
    if show_method_labels:
        ax_label = fig.add_subplot(gs[2, 0])
        ax_label.axis("off")
        ax_label.text(
            0.5,
            0.5,
            first_label,
            ha="center",
            va="center",
            fontsize=row_label_fontsize,
            fontweight="bold",
            linespacing=1.25,
        )
    ref_ax = _draw_candidate_row(
        fig,
        gs[2, candidates_col],
        first_imgs,
        first_flags,
        top_k,
    )
    _draw_rank_labels(
        fig,
        gs[1, candidates_col],
        top_k=top_k,
        tile_width=tile_width,
        x_reference_ax=ref_ax,
        rank_fontsize=rank_fontsize,
    )

    for row_idx, (row_label, cand_imgs, cand_flags) in enumerate(candidate_rows[1:], start=1):
        grid_row = row_idx + 2
        if show_method_labels:
            ax_label = fig.add_subplot(gs[grid_row, 0])
            ax_label.axis("off")
            ax_label.text(
                0.5,
                0.5,
                row_label,
                ha="center",
                va="center",
                fontsize=row_label_fontsize,
                fontweight="bold",
                linespacing=1.25,
            )
        _draw_candidate_row(
            fig,
            gs[grid_row, candidates_col],
            cand_imgs,
            cand_flags,
            top_k,
            sharex=ref_ax,
        )


def visualize_methods(
    *,
    method_rows: list[tuple[str, Path]] = METHOD_ROWS,
    query_meta_path: Path = QUERY_META,
    db_meta_path: Path = DB_META,
    output_path: Path = DEFAULT_OUTPUT,
    query_id: int | None = None,
    query_id_right: int | None = None,
    dual_panel: bool = True,
    top_k: int = 5,
    dpi: int = 150,
) -> tuple[int, int | None]:
    q_df = pd.read_parquet(query_meta_path)
    db_df = pd.read_parquet(db_meta_path)

    if query_id is None:
        query_id = find_primary_query(method_rows, top_k=top_k)

    left_rows = _load_panel_rows(
        query_id=query_id,
        method_rows=method_rows,
        db_df=db_df,
        top_k=top_k,
    )

    right_query_id: int | None = None
    right_rows: list[tuple[str, list[np.ndarray], list[bool]]] | None = None
    if dual_panel:
        if query_id_right is None:
            query_id_right = find_secondary_query(method_rows, top_k=top_k)
        right_query_id = query_id_right
        right_rows = _load_panel_rows(
            query_id=right_query_id,
            method_rows=method_rows,
            db_df=db_df,
            top_k=top_k,
        )

    n_rows = len(left_rows)
    rank_extra = _rank_label_extra_height_ratio(METHOD_RANK_FONTSIZE)
    panel_height = 2.5 * (n_rows + 1 + rank_extra)
    if dual_panel:
        dual_fig_w = SINGLE_PANEL_FIG_W * (LEFT_PANEL_RATIO + 0.02 + CAND_COL_RATIO) / LEFT_PANEL_RATIO
        fig = plt.figure(figsize=(dual_fig_w, panel_height), facecolor="white")
        outer = fig.add_gridspec(
            1,
            3,
            width_ratios=[LEFT_PANEL_RATIO, 0.02, CAND_COL_RATIO],
            wspace=0.05,
        )
        _draw_methods_panel(
            fig,
            outer[0, 0],
            query_id=query_id,
            q_df=q_df,
            candidate_rows=left_rows,
            top_k=top_k,
        )
        ax_div = fig.add_subplot(outer[0, 1])
        ax_div.set_xlim(0, 1)
        ax_div.set_ylim(0, 1)
        ax_div.axis("off")
        ax_div.axvline(0.5, color="black", linewidth=2.0)
        _draw_methods_panel(
            fig,
            outer[0, 2],
            query_id=right_query_id,
            q_df=q_df,
            candidate_rows=right_rows,
            top_k=top_k,
            show_method_labels=False,
        )
    else:
        fig = plt.figure(figsize=(SINGLE_PANEL_FIG_W, panel_height), facecolor="white")
        outer = fig.add_gridspec(1, 1)
        _draw_methods_panel(
            fig,
            outer[0, 0],
            query_id=query_id,
            q_df=q_df,
            candidate_rows=left_rows,
            top_k=top_k,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Selected query_id={query_id}")
    if dual_panel:
        print(f"Selected right query_id={right_query_id}")
    print(f"Saved visualization to {output_path}")
    return query_id, right_query_id


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--query-meta", type=Path, default=QUERY_META)
    p.add_argument("--db-meta", type=Path, default=DB_META)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--query-id", type=int, default=None)
    p.add_argument("--query-id-right", type=int, default=None)
    p.add_argument("--single-panel", action="store_true", help="Disable the right-side panel")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--dpi", type=int, default=150)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    visualize_methods(
        query_meta_path=args.query_meta.resolve(),
        db_meta_path=args.db_meta.resolve(),
        output_path=args.output.resolve(),
        query_id=args.query_id,
        query_id_right=args.query_id_right,
        dual_panel=not args.single_panel,
        top_k=args.top_k,
        dpi=args.dpi,
    )


if __name__ == "__main__":
    main()
