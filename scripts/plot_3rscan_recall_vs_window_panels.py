#!/usr/bin/env python3
"""Recall@k vs window size on 3RScan: 3x3 grid (sim conditions x recall@k)."""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt

SCRIPTS = Path(__file__).resolve().parent
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from plot_methods_recall_vs_window import _window_from_dirname

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = ROOT / "data" / "tests" / "26-06-01" / "3RScan"
DEFAULT_OUTPUT = ROOT / "data" / "plots" / "3rscan_recall_vs_window_panels.png"
REPORT = "base_seq_report"

METHODS: list[tuple[str, str]] = [
    ("EDTFormer", "EDTformer/rerank_k_500_per_frame_k_25"),
    ("FoL", "FoL_base/rerank_k_50_per_frame_k_25"),
    ("MegaLoc", "Megaloc/rerank_k_500_per_frame_k_25"),
    ("SelaVPR++", "SelaVPRpp/rerank_k_100_per_frame_k_25"),
    ("GraphSeqLoc (GT)", "GT/64xMegaloc/rerank_k_1000_per_frame_k_25"),
    ("GraphSeqLoc (SceneGraphVLM)", "Makarov/64xMegaloc/rerank_k_1000_per_frame_k_25"),
]

ROWS: list[tuple[str, str]] = [
    ("room-sim", "room-sim"),
    ("pose-far-sim", "3 m condition"),
    ("pose-near-sim", "2 m condition"),
]

# Optional manual y-axis range per panel: key "subdir@k", clipped to [0, 1].
PANEL_YLIM: dict[str, tuple[float, float]] = {}

COLS: list[tuple[int, str]] = [
    (1, "Recall@1"),
    (5, "Recall@5"),
    (10, "Recall@10"),
]

COLORS = {
    "EDTFormer": "#1f77b4",
    "FoL": "#d62728",
    "MegaLoc": "#ff7f0e",
    "SelaVPR++": "#2ca02c",
    "GraphSeqLoc (GT)": "#9467bd",
    "GraphSeqLoc (SceneGraphVLM)": "#8c564b",
}

FIGSIZE = (11.0, 11.0)
FONT_TICK = 10
FONT_LABEL = 11
FONT_TITLE = 11
FONT_SUPTITLE = 12
FONT_LEGEND = 10
LINE_WIDTH = 2.0
MARKER_SIZE = 5.0
ERROR_CAP = 3.5


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help=f"3RScan test directory (default: {DEFAULT_DATASET_ROOT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Do not draw std error bars from metrics.json",
    )
    parser.add_argument(
        "--ylim",
        action="append",
        metavar="PANEL:MIN,MAX",
        help=(
            "Override visible y-axis range for one panel, e.g. pose-near-sim@1:0.30,0.42. "
            "PANEL is subdir@k (room-sim@1, pose-far-sim@5, pose-near-sim@10, ...)."
        ),
    )
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive plot window")
    return parser.parse_args()


def panel_key(subdir: str, recall_k: int) -> str:
    return f"{subdir}@{recall_k}"


def parse_ylim_overrides(raw: list[str] | None) -> dict[str, tuple[float, float]]:
    overrides: dict[str, tuple[float, float]] = {}
    if not raw:
        return overrides
    for item in raw:
        panel_name, limits = item.split(":", 1)
        ymin_s, ymax_s = limits.split(",", 1)
        overrides[panel_name] = (float(ymin_s), float(ymax_s))
    return overrides


def load_recall_curve_at_k(
    method_dir: Path,
    *,
    report: str,
    subdir: str,
    k: int,
) -> tuple[list[int], list[float], list[float] | None]:
    """Load Recall@k vs window from metrics.json files."""
    curve_dir = method_dir / report / subdir
    if not curve_dir.is_dir():
        raise FileNotFoundError(f"Metrics subdirectory not found: {curve_dir}")

    windows: list[int] = []
    recall: list[float] = []
    std: list[float] = []
    k_key = str(k)

    for window_path in sorted(curve_dir.iterdir()):
        if not window_path.is_dir():
            continue
        metrics_path = window_path / "metrics.json"
        if not metrics_path.is_file():
            continue

        payload = json.loads(metrics_path.read_text())
        w = payload.get("config", {}).get("max_window")
        if w is None:
            w = _window_from_dirname(window_path.name)
        if w is None:
            continue

        recall_at_k = payload.get("recall_at_k") or {}
        rk = recall_at_k.get(k_key) or recall_at_k.get(k)
        if rk is None:
            continue

        windows.append(int(w))
        recall.append(float(rk))

        recall_std = payload.get("recall_at_k_std") or {}
        sk = recall_std.get(k_key) or recall_std.get(k)
        std.append(float(sk) if sk is not None else 0.0)

    if not windows:
        raise FileNotFoundError(
            f"No window metrics found under {curve_dir} (expected *-window/metrics.json)"
        )

    order = sorted(range(len(windows)), key=lambda i: windows[i])
    windows_sorted = [windows[i] for i in order]
    recall_sorted = [recall[i] for i in order]
    std_sorted = [std[i] for i in order]
    has_std = any(s > 0 for s in std_sorted)
    return windows_sorted, recall_sorted, std_sorted if has_std else None


def compute_panel_ylim(
    dataset_root: Path,
    subdir: str,
    recall_k: int,
    *,
    show_std: bool,
    pad_frac: float = 0.08,
    min_pad: float = 0.015,
) -> tuple[float, float]:
    """Pick a tight y-range for one panel, clipped to [0, 1]."""
    ymin = math.inf
    ymax = -math.inf

    for _, method_rel in METHODS:
        method_dir = dataset_root / method_rel
        _, recall, std = load_recall_curve_at_k(
            method_dir,
            report=REPORT,
            subdir=subdir,
            k=recall_k,
        )
        for i, value in enumerate(recall):
            err = std[i] if show_std and std is not None else 0.0
            ymin = min(ymin, value - err)
            ymax = max(ymax, value + err)

    span = ymax - ymin
    pad = max(min_pad, span * pad_frac)
    return max(0.0, ymin - pad), min(1.0, ymax + pad)


def panel_ylim(
    dataset_root: Path,
    subdir: str,
    recall_k: int,
    *,
    show_std: bool,
    overrides: dict[str, tuple[float, float]],
) -> tuple[float, float]:
    key = panel_key(subdir, recall_k)
    if key in overrides:
        ymin, ymax = overrides[key]
    elif key in PANEL_YLIM:
        ymin, ymax = PANEL_YLIM[key]
    else:
        ymin, ymax = compute_panel_ylim(
            dataset_root,
            subdir,
            recall_k,
            show_std=show_std,
        )
    return max(0.0, ymin), min(1.0, ymax)


def plot_panel(
    ax: plt.Axes,
    *,
    dataset_root: Path,
    subdir: str,
    recall_k: int,
    show_std: bool,
) -> None:
    ax.set_axisbelow(True)

    for label, method_rel in METHODS:
        method_dir = dataset_root / method_rel
        windows, recall, std = load_recall_curve_at_k(
            method_dir,
            report=REPORT,
            subdir=subdir,
            k=recall_k,
        )
        color = COLORS[label]
        kwargs = {
            "marker": "o",
            "markersize": MARKER_SIZE,
            "linewidth": LINE_WIDTH,
            "label": label,
            "color": color,
        }
        if show_std and std is not None:
            ax.errorbar(windows, recall, yerr=std, capsize=ERROR_CAP, **kwargs)
        else:
            ax.plot(windows, recall, **kwargs)

    ax.grid(True, alpha=0.3)


def main() -> None:
    args = parse_args()
    dataset_root = args.dataset_root.resolve()
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset path not found: {dataset_root}")

    show_std = not args.no_std
    ylim_overrides = parse_ylim_overrides(args.ylim)

    fig, axes = plt.subplots(
        len(ROWS),
        len(COLS),
        figsize=FIGSIZE,
        sharex=True,
    )

    for row_idx, (subdir, row_title) in enumerate(ROWS):
        for col_idx, (recall_k, col_title) in enumerate(COLS):
            ax = axes[row_idx, col_idx]
            plot_panel(
                ax,
                dataset_root=dataset_root,
                subdir=subdir,
                recall_k=recall_k,
                show_std=show_std,
            )
            ymin, ymax = panel_ylim(
                dataset_root,
                subdir,
                recall_k,
                show_std=show_std,
                overrides=ylim_overrides,
            )
            ax.set_ylim(ymin, ymax)
            ax.margins(y=0)
            ax.tick_params(labelsize=FONT_TICK)

            if row_idx == 0:
                ax.set_title(col_title, fontsize=FONT_TITLE)
            if col_idx == 0:
                ax.set_ylabel(f"{row_title}\nRecall", fontsize=FONT_LABEL)
            if row_idx == len(ROWS) - 1:
                ax.set_xlabel("Sequence window size", fontsize=FONT_LABEL)
    
            print(f"{row_title} / Recall@{recall_k}: y-axis [{ymin:.3f}, {ymax:.3f}]")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    # fig.suptitle("3RScan", fontsize=FONT_SUPTITLE)
    fig.tight_layout(rect=(0, 0.09, 1, 0.98))

    bottom_row = axes[-1, 1].get_position()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.08),
        ncol=3,
        fontsize=FONT_LEGEND,
        frameon=False,
    )

    out = args.output if args.output.is_absolute() else ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.05)
    print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
