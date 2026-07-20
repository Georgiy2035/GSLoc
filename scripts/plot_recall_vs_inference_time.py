#!/usr/bin/env python3
"""Scatter plot: model retrieval time vs Recall@1 (room-sim, 001-window) on 3RScan_BIG."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_ROOT = ROOT / "data" / "tests" / "26-05-14" / "3RScan_BIG"
DEFAULT_OUTPUT = ROOT / "data" / "plots" / "3rscan_big_recall_vs_inference_time.png"

METHODS: list[tuple[str, str]] = [
    ("EDTFormer", "EDTformer/rerank_k_500_per_frame_k_25"),
    ("MegaLoc", "Megaloc/rerank_k_500_per_frame_k_25"),
    ("SelaVPR++", "SelaVPRpp/rerank_k_500_per_frame_k_25"),
    ("FoL", "Fol_base/rerank_k_500_per_frame_k_25"),
    ("GraphSeqLoc", "GT/64xMegaloc/rerank_k_1000_per_frame_k_25"),
]

COLORS = {
    "EDTFormer": "#1f77b4",
    "MegaLoc": "#ff7f0e",
    "SelaVPR++": "#2ca02c",
    "FoL": "#d62728",
    "GraphSeqLoc": "#9467bd",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help=f"Test run directory (default: {DEFAULT_DATA_ROOT})",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output PNG path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window",
    )
    return parser.parse_args()


def load_time_ms(time_path: Path) -> float:
    with time_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    model_infer_s = payload["full_inference_time"]
    return model_infer_s * 1000.0


def load_recall_at_1(metrics_path: Path) -> float:
    with metrics_path.open(encoding="utf-8") as f:
        payload = json.load(f)
    recall_at_k = payload.get("recall_at_k") or {}
    value = recall_at_k.get("1") or recall_at_k.get(1)
    if value is None:
        raise KeyError(f"recall_at_k['1'] not found in {metrics_path}")
    return float(value)


def load_gsloc_recall(method_dir: Path) -> tuple[float, str]:
    room_sim_dir = method_dir / "base_seq_report" / "room-sim"
    if not room_sim_dir.is_dir():
        raise FileNotFoundError(f"room-sim report not found: {room_sim_dir}")

    best_recall = float("-inf")
    best_window = ""
    for window_dir in sorted(room_sim_dir.glob("*-window")):
        metrics_path = window_dir / "metrics.json"
        if not metrics_path.is_file():
            continue
        recall = load_recall_at_1(metrics_path)
        if recall > best_recall:
            best_recall = recall
            best_window = window_dir.name

    if best_window == "":
        raise FileNotFoundError(f"No metrics.json files found under {room_sim_dir}")

    return best_recall, best_window


def collect_points(data_root: Path) -> list[dict[str, object]]:
    points: list[dict[str, object]] = []

    for label, rel_path in METHODS:
        method_dir = data_root / rel_path
        time_path = method_dir / "time.json"
        if not time_path.is_file():
            raise FileNotFoundError(f"time.json not found: {time_path}")

        time_ms = load_time_ms(time_path)

        if label == "GraphSeqLoc":
            recall, recall_window = load_gsloc_recall(method_dir)
            recall_source = f"max room-sim ({recall_window})"
        else:
            metrics_path = method_dir / "base_seq_report" / "room-sim" / "001-window" / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"metrics.json not found: {metrics_path}")
            recall = load_recall_at_1(metrics_path)
            recall_source = "room-sim 001-window"

        points.append(
            {
                "label": label,
                "time_ms": time_ms,
                "recall_pct": recall * 100.0,
                "recall_source": recall_source,
            }
        )

    return points


def build_plot(points: list[dict[str, object]], *, dataset: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(4, 3))
    ax.set_axisbelow(True)

    for point in points:
        label = str(point["label"])
        x = float(point["time_ms"])
        y = float(point["recall_pct"])
        color = COLORS.get(label, "#333333")
        ax.scatter(
            x,
            y,
            s=110,
            color=color,
            edgecolors="k",
            linewidths=0.6,
            zorder=3,
            label=label,
        )
        ax.annotate(
            label,
            (x, y),
            xytext=(8, 6),
            textcoords="offset points",
            fontsize=9,
            ha="left",
            va="bottom",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.85", alpha=0.95),
            arrowprops=dict(arrowstyle="-", color="0.45", lw=0.5),
        )

    ax.set_xlabel("Model retrieval time (ms)")
    ax.set_ylabel("Recall@1 (%)")
    #ax.set_title(f"{dataset}, room-sim")
    ax.grid(True, alpha=0.3)
    #ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    data_root = args.data_root.resolve()
    if not data_root.is_dir():
        raise SystemExit(f"Data directory not found: {data_root}")

    points = collect_points(data_root)
    for point in points:
        print(
            f"{point['label']}: "
            f"R@1={point['recall_pct']:.2f}% ({point['recall_source']}), "
            f"time={point['time_ms']:.2f} ms"
        )

    fig = build_plot(points, dataset=data_root.name)

    output = args.output if args.output.is_absolute() else ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    print(f"Saved: {output}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
