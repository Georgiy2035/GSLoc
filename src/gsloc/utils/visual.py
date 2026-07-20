import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import pandas as pd
from pathlib import Path

def plot_metrics_vs_window_with_stats(
    summary_df,
    summary_all,
    metrics = ("auc_pr", "f1_max", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_25"),
):
    """Plot per-map metrics vs w and overlay cross-map mean and weighted mean.

    Args:
        summary_df: DataFrame for a single map (has columns 'w', metrics, and optional '<metric>_std').
        summary_all: Concatenated DataFrame across maps with columns 'w', 'query_track', 'num_valid', and metrics.
        metrics: metric names to visualize.
    Returns:
        dict metric -> plotly figure
    """
    figs = {}
    df = summary_df.sort_values("w").reset_index(drop=True)
    map_name = "all"

    for m in metrics:
        if m not in df.columns:
            continue

        std_col = f"{m}_std"
        has_std = std_col in df.columns

        line_kwargs = {
            "x": "w",
            "y": m,
            "title": f"{map_name}: {m} vs sequence length (w)",
            "markers": True,
        }
        if has_std:
            line_kwargs["error_y"] = std_col

        fig = px.line(df, **line_kwargs)
        fig.update_layout(xaxis_title="sequence length (max_window)", yaxis_title=m)

        # Highlight maximum point on per-map line
        try:
            idx_max = df[m].astype(float).idxmax()
            w_star = int(df.loc[idx_max, "w"])  # sequence length at max
            y_star = float(df.loc[idx_max, m])
            fig.add_trace(
                go.Scatter(x=[w_star], y=[y_star], mode="markers", marker=dict(color="red", size=10), name="max", showlegend=False)
            )
            try:
                fig.add_vline(x=w_star, line_dash="dash", line_color="red")
            except Exception:
                fig.add_shape(type="line", x0=w_star, x1=w_star, y0=min(df[m].astype(float)), y1=max(df[m].astype(float)), line=dict(color="red", dash="dash"))
            fig.add_annotation(x=w_star, y=y_star, text=f"w={w_star}, {m}={y_star:.4f}", showarrow=True, arrowhead=2, ax=40, ay=-40)
        except Exception:
            pass

        # Overlay weighted mean across maps (weights = num_valid per map)
        try:
            wmean_series = (
                summary_all
                .groupby("w")
                .apply(lambda g: float(np.average(g[m].astype(float), weights=g["num_valid"].astype(float))), include_groups=False)
                .reset_index(name=m)
            )

            weighted_mean_kwargs = {
                "x": wmean_series["w"],
                "y": wmean_series[m].astype(float),
                "mode": "lines",
                "name": "weighted mean",
                "line": dict(color="purple", dash="dot"),
                "showlegend": True,
            }

            if std_col in summary_all.columns:
                wstd_series = (
                    summary_all
                    .groupby("w")
                    .apply(lambda g: float(np.average(g[std_col].astype(float), weights=g["num_valid"].astype(float))), include_groups=False)
                    .reset_index(name=std_col)
                )
                wmean_series = wmean_series.merge(wstd_series, on="w", how="left")
                weighted_mean_kwargs["error_y"] = dict(type="data", array=wmean_series[std_col].astype(float), visible=True)

            fig.add_trace(go.Scatter(**weighted_mean_kwargs))
        except Exception:
            pass

        figs[m] = fig
        fig.show()
    return figs


def plot_metrics_from_parquet(
    summary_df_path,
    summary_all_path=None,
    metrics=("auc_pr", "f1_max", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_25"),
):
    """Read parquet report(s) and plot metrics vs window.

    Args:
        summary_df_path: Path to parquet with per-window metrics.
        summary_all_path: Optional parquet path with concatenated metrics across maps.
            If omitted, `summary_df_path` is reused.
        metrics: Metric names to visualize.
    Returns:
        dict metric -> plotly figure
    """
    summary_df = pd.read_parquet(Path(summary_df_path))
    if summary_all_path is None:
        summary_all = summary_df.copy()
    else:
        summary_all = pd.read_parquet(Path(summary_all_path))

    if "num_valid" not in summary_all.columns:
        # Fallback to unweighted behavior in weighted overlays.
        summary_all = summary_all.copy()
        summary_all["num_valid"] = 1.0

    return plot_metrics_vs_window_with_stats(
        summary_df=summary_df,
        summary_all=summary_all,
        metrics=metrics,
    )


def plot_metrics_from_experiment_dir(
    experiment_dir,
    metrics=("auc_pr", "f1_max", "recall_at_1", "recall_at_5", "recall_at_10", "recall_at_25"),
    parquet_name="summaryresults.parquet",
):
    """Find all parquet summaries in directory tree and plot each one.

    Args:
        experiment_dir: Root directory to scan recursively.
        metrics: Metric names to visualize.
        parquet_name: File name to search for.
    Returns:
        dict[str, dict]: mapping "parquet_path" -> {metric: figure}
    """
    root = Path(experiment_dir)
    parquet_paths = sorted(root.rglob(parquet_name))
    if not parquet_paths:
        raise FileNotFoundError(f"No '{parquet_name}' found under: {root}")

    summary_all = pd.concat(
        [pd.read_parquet(path).assign(_source=str(path)) for path in parquet_paths],
        ignore_index=True,
    )
    if "num_valid" not in summary_all.columns:
        summary_all = summary_all.copy()
        summary_all["num_valid"] = 1.0

    result = {}
    for path in parquet_paths:
        summary_df = pd.read_parquet(path)
        result[str(path)] = plot_metrics_vs_window_with_stats(
            summary_df=summary_df,
            summary_all=summary_all,
            metrics=metrics,
        )
    return result
