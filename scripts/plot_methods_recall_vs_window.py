#!/usr/bin/env python3
"""Compare Recall@1 vs window size for several methods on one test dataset."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TESTS_ROOT = ROOT / "data" / "tests"
WINDOW_DIR_RE = re.compile(r"^(\d+)-window$")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot Recall@1 vs sequence window for several methods on one dataset. "
            "Metrics are read from metrics.json under each method's report subdirectory."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --date 26-06-01 --dataset 3RScan --subdir room-sim \\
    --methods SelaVPRpp/rerank_k_100_per_frame_k_25 Fross/64xMegaloc/rerank_k_1000_per_frame_k_25 \\
    --output data/plots/3rscan_room_sim.png

  %(prog)s --list-dates
  %(prog)s --date 26-06-01 --list-datasets
  %(prog)s --date 26-06-01 --dataset 3RScan --list-methods
""",
    )
    parser.add_argument(
        "--tests-root",
        type=Path,
        default=DEFAULT_TESTS_ROOT,
        help=f"Root directory with test runs (default: {DEFAULT_TESTS_ROOT})",
    )
    parser.add_argument("--date", help="Test run date folder, e.g. 26-06-01")
    parser.add_argument("--dataset", help="Dataset name under the date folder, e.g. 3RScan")
    parser.add_argument(
        "--subdir",
        help="Similarity subdirectory under the report, e.g. room-sim, pose-far-sim",
    )
    parser.add_argument(
        "--report",
        default="base_seq_report",
        help="Report directory name (default: base_seq_report)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        metavar="METHOD",
        help=(
            "Method paths relative to {tests-root}/{date}/{dataset}/, "
            "e.g. SelaVPRpp/rerank_k_100_per_frame_k_25"
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        metavar="LABEL",
        help="Legend labels (same count as --methods; default: method folder name)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        help="Save plot to this path (also shown interactively unless --no-show)",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Show Recall@1 on 0–100 scale instead of 0–1",
    )
    parser.add_argument(
        "--no-std",
        action="store_true",
        help="Do not draw std error bars from metrics.json",
    )
    parser.add_argument("--title", help="Custom plot title")
    parser.add_argument("--no-show", action="store_true", help="Do not open interactive plot window")
    parser.add_argument("--list-dates", action="store_true", help="List available date folders and exit")
    parser.add_argument(
        "--list-datasets",
        action="store_true",
        help="List datasets for --date and exit (requires --date)",
    )
    parser.add_argument(
        "--list-methods",
        action="store_true",
        help="List method configs for --date/--dataset and exit (requires both)",
    )
    return parser.parse_args()


def list_dates(tests_root: Path) -> list[str]:
    if not tests_root.is_dir():
        raise FileNotFoundError(f"Tests root not found: {tests_root}")
    return sorted(p.name for p in tests_root.iterdir() if p.is_dir())


def list_datasets(tests_root: Path, date: str) -> list[str]:
    dataset_root = tests_root / date
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Date folder not found: {dataset_root}")
    return sorted(
        p.name
        for p in dataset_root.iterdir()
        if p.is_dir() and p.name != "cache"
    )


def list_methods(tests_root: Path, date: str, dataset: str, report: str) -> list[str]:
    """Return relative method paths that contain at least one metrics.json."""
    dataset_root = tests_root / date / dataset
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_root}")

    found: list[str] = []
    for metrics_path in sorted(dataset_root.rglob("metrics.json")):
        try:
            rel = metrics_path.relative_to(dataset_root)
        except ValueError:
            continue
        parts = rel.parts
        if report not in parts:
            continue
        report_idx = parts.index(report)
        if report_idx < 1:
            continue
        method_rel = "/".join(parts[:report_idx])
        if method_rel and method_rel not in found:
            found.append(method_rel)
    return found


def _window_from_dirname(name: str) -> int | None:
    match = WINDOW_DIR_RE.match(name)
    return int(match.group(1)) if match else None


def load_recall_curve(
    method_dir: Path,
    *,
    report: str,
    subdir: str,
) -> tuple[list[int], list[float], list[float] | None]:
    """Load Recall@1 vs window from metrics.json files."""
    curve_dir = method_dir / report / subdir
    if not curve_dir.is_dir():
        raise FileNotFoundError(f"Metrics subdirectory not found: {curve_dir}")

    windows: list[int] = []
    recall: list[float] = []
    std: list[float] = []

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
        r1 = recall_at_k.get("1") or recall_at_k.get(1)
        if r1 is None:
            continue

        windows.append(int(w))
        recall.append(float(r1))

        recall_std = payload.get("recall_at_k_std") or {}
        s1 = recall_std.get("1") or recall_std.get(1)
        std.append(float(s1) if s1 is not None else 0.0)

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


METHOD_DISPLAY_NAMES: dict[str, str] = {
    "EDTformer": "EDTFormer",
    "FoL_base": "FoL",
    "Fol_base": "FoL",
    "Megaloc": "MegaLoc",
    "SelaVPRpp": "SelaVPR++",
    "GT/64xMegaloc": "GraphSeqLoc (GT)",
    "Makarov/64xMegaloc": "GraphSeqLoc (SceneGraphVLM)",
    "Makarov/64_pure": "GraphSeqLoc (64_pure)",
    "Fross/64xMegaloc": "Fross",
}


def default_label(method: str) -> str:
    rel = method.rstrip("/")
    parts = rel.split("/")
    for end in range(len(parts), 0, -1):
        key = "/".join(parts[:end])
        if key in METHOD_DISPLAY_NAMES:
            return METHOD_DISPLAY_NAMES[key]
    if parts and parts[-1].startswith("rerank_"):
        return "/".join(parts[:-1])
    return parts[-1]


def build_plot(
    series: list[tuple[str, list[int], list[float], list[float] | None]],
    *,
    date: str,
    dataset: str,
    subdir: str,
    percent: bool,
    show_std: bool,
    title: str | None,
) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.set_axisbelow(True)

    scale = 100.0 if percent else 1.0
    y_label = "Recall@1 (%)" if percent else "Recall@1"

    for label, windows, recall, std in series:
        y = [v * scale for v in recall]
        kwargs = {"marker": "o", "linewidth": 1.8, "label": label}
        if show_std and std is not None:
            yerr = [v * scale for v in std]
            ax.errorbar(windows, y, yerr=yerr, capsize=3, **kwargs)
        else:
            ax.plot(windows, y, **kwargs)

    ax.set_xlabel("Sequence window size")
    ax.set_ylabel(y_label)
    if title:
        ax.set_title(title)
    else:
        ax.set_title(f"{dataset}, {subdir}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def main() -> None:
    args = parse_args()
    tests_root = args.tests_root.resolve()

    if args.list_dates:
        for name in list_dates(tests_root):
            print(name)
        return

    if args.list_datasets:
        if not args.date:
            raise SystemExit("--list-datasets requires --date")
        for name in list_datasets(tests_root, args.date):
            print(name)
        return

    if args.list_methods:
        if not args.date or not args.dataset:
            raise SystemExit("--list-methods requires --date and --dataset")
        for name in list_methods(tests_root, args.date, args.dataset, args.report):
            print(name)
        return

    missing = [
        flag
        for flag, value in (
            ("--date", args.date),
            ("--dataset", args.dataset),
            ("--subdir", args.subdir),
            ("--methods", args.methods),
        )
        if not value
    ]
    if missing:
        raise SystemExit(f"Required arguments missing: {', '.join(missing)}")

    if args.labels and len(args.labels) != len(args.methods):
        raise SystemExit("--labels count must match --methods count")

    dataset_root = tests_root / args.date / args.dataset
    if not dataset_root.is_dir():
        raise SystemExit(f"Dataset path not found: {dataset_root}")

    labels = args.labels or [default_label(m) for m in args.methods]
    series: list[tuple[str, list[int], list[float], list[float] | None]] = []

    for method, label in zip(args.methods, labels, strict=True):
        method_dir = dataset_root / method
        if not method_dir.is_dir():
            raise SystemExit(f"Method directory not found: {method_dir}")
        windows, recall, std = load_recall_curve(
            method_dir,
            report=args.report,
            subdir=args.subdir,
        )
        series.append((label, windows, recall, std))
        print(
            f"{label}: {len(windows)} points, "
            f"windows={windows[0]}..{windows[-1]}, "
            f"R@1={recall[-1] * (100 if args.percent else 1):.2f}"
            f"{'%' if args.percent else ''} @ w={windows[-1]}"
        )

    fig = build_plot(
        series,
        date=args.date,
        dataset=args.dataset,
        subdir=args.subdir,
        percent=args.percent,
        show_std=not args.no_std,
        title=args.title,
    )

    if args.output:
        out = args.output if args.output.is_absolute() else ROOT / args.output
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"Saved: {out}")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
