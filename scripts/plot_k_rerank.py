#!/usr/bin/env python3
"""Scatter plot: inference time (ms) vs Recall@1 from data/k_rerank.csv."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "k_rerank.csv"
OUT_PATH = ROOT / "data" / "k_rerank_r1_vs_time.png"

K_SERIES = {"50", "100", "250", "500", "750", "1500", "2000", "1000 (64g→ML)"}
COL_LEFT = {"50", "100", "250", "500", "FoL_base"}
COL_RIGHT = {"750", "MegaLoc", "1000 (64g→ML)", "1500", "2000"}

COL_X_LEFT = 35.2
COL_X_RIGHT = 54.5
GAP_LEFT, GAP_RIGHT = 1.12, 1.38

TEXT_KW = dict(
    fontsize=8,
    ha="left",
    va="center",
    zorder=5,
    bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="0.85", alpha=0.95),
)
ARROW_KW = dict(arrowstyle="-", color="0.45", lw=0.45, shrinkA=2, shrinkB=1)


def parse_r1(value: str) -> float:
    return float(str(value).strip().replace("%", ""))


def parse_time_ms(value: str) -> float | None:
    s = str(value).strip().replace(",", ".")
    if re.fullmatch(r"[\d.]+", s):
        return float(s)
    return None


def point_label(row: pd.Series) -> str:
    base = str(row["Основная модель"]).strip()
    rerank = str(row["Модель реранж-ния"]).strip()
    rk = row["rerank_k"]
    has_k = pd.notna(rk) and str(rk).strip() != "-"

    if not has_k:
        return base

    k = str(int(rk)) if float(rk) == int(float(rk)) else str(rk)

    if k == "1000":
        if base == "64_graph" and rerank == "MegaLoc":
            return "1000 (64g→ML)"
        if base == "MegaLoc" and rerank == "64_graph":
            return "1000 (ML→64g)"

    if base == "64_graph" and rerank == "MegaLoc":
        return k

    return f"{base}+{rerank} {k}"


def spread_labels(y_targets: np.ndarray, min_gap: float) -> np.ndarray:
    """Minimum vertical spacing with backward pass to stay near points."""
    y = np.asarray(y_targets, dtype=float)
    order = np.argsort(y)
    ys = y[order].copy()
    for i in range(1, len(ys)):
        if ys[i] - ys[i - 1] < min_gap:
            ys[i] = ys[i - 1] + min_gap
    for i in range(len(ys) - 2, -1, -1):
        if ys[i + 1] - ys[i] < min_gap:
            ys[i] = ys[i + 1] - min_gap
    out = np.empty_like(y)
    out[order] = ys
    return out


def main() -> None:
    df = pd.read_csv(CSV_PATH, sep="\t")
    df["R@1_val"] = df["R@1"].map(parse_r1)
    df["time_ms"] = df["Время (мс)"].map(parse_time_ms)
    df["label"] = df.apply(point_label, axis=1)
    plot_df = df.dropna(subset=["time_ms"]).copy()

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    ax.set_axisbelow(True)
    ax.scatter(
        plot_df["time_ms"],
        plot_df["R@1_val"],
        s=80,
        alpha=0.9,
        edgecolors="k",
        linewidths=0.5,
        zorder=3,
    )

    columns = [
        (COL_LEFT, COL_X_LEFT, "right", GAP_LEFT),
        (COL_RIGHT, COL_X_RIGHT, "left", GAP_RIGHT),
    ]
    for col_labels, col_x, ha, gap in columns:
        col_df = plot_df[plot_df["label"].isin(col_labels)].sort_values("R@1_val")
        if col_df.empty:
            continue
        col_y = spread_labels(col_df["R@1_val"].to_numpy(), gap)
        for (_, row), y_text in zip(col_df.iterrows(), col_y):
            x, y = row["time_ms"], row["R@1_val"]
            ax.annotate(
                row["label"],
                (x, y),
                xytext=(col_x, y_text),
                textcoords="data",
                ha=ha,
                va="center",
                fontsize=8,
                zorder=5,
                bbox=TEXT_KW["bbox"],
                arrowprops=ARROW_KW,
            )

    col_mask = plot_df["label"].isin(COL_LEFT | COL_RIGHT)

    # Isolated points — local offsets
    local = {
        "64_graph": (1.8, 2.2, "left"),
        "1000 (ML→64g)": (1.2, -1.8, "left"),
    }
    for _, row in plot_df[~col_mask].iterrows():
        label = row["label"]
        x, y = row["time_ms"], row["R@1_val"]
        dx, dy, ha = local.get(label, (0.6, 0.4, "left"))
        ax.annotate(
            label,
            (x, y),
            xytext=(x + dx, y + dy),
            textcoords="data",
            fontsize=8,
            ha=ha,
            va="center",
            zorder=5,
            bbox=TEXT_KW["bbox"],
            arrowprops=ARROW_KW,
        )

    x_min, x_max = plot_df["time_ms"].min(), plot_df["time_ms"].max()
    y_min, y_max = plot_df["R@1_val"].min(), plot_df["R@1_val"].max()
    col_all = plot_df[col_mask]
    y_top = col_all["R@1_val"].max() + 3 if not col_all.empty else y_max + 2
    ax.set_xlim(x_min - 2, COL_X_RIGHT + 2.5)
    ax.set_ylim(y_min - 4, y_top)

    ax.set_xlabel("Время (мс)")
    ax.set_ylabel("R@1 (%)")
    ax.set_title("Recall@1 vs время инференса (k_rerank)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
