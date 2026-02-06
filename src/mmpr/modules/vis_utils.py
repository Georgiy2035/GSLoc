from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from tf_math import Transform


def plot_alignment(
    db_map_pcd,
    minimap_pts,
    query_pts_init,
    query_pts_refined,
    info_text=None,
    bounds_margin=2.0,
):
    """Plot base map (cropped), mini-map, and query before/after refinement in XY."""
    # Compute focused bounds
    stack = [minimap_pts[:, :2]]
    if query_pts_init is not None:
        stack.append(query_pts_init[:, :2])
    if query_pts_refined is not None:
        stack.append(query_pts_refined[:, :2])
    all_xy = np.vstack(stack)
    x_min, x_max = (
        all_xy[:, 0].min() - bounds_margin,
        all_xy[:, 0].max() + bounds_margin,
    )
    y_min, y_max = (
        all_xy[:, 1].min() - bounds_margin,
        all_xy[:, 1].max() + bounds_margin,
    )

    # Filter base map to focused region
    mask = (
        (db_map_pcd[:, 0] >= x_min)
        & (db_map_pcd[:, 0] <= x_max)
        & (db_map_pcd[:, 1] >= y_min)
        & (db_map_pcd[:, 1] <= y_max)
    )
    db_crop = db_map_pcd[mask]

    # Plot
    fig, ax = plt.subplots(figsize=(18, 16))
    ax.scatter(db_crop[:, 0], db_crop[:, 1], s=1, alpha=1, label="DB map (crop)")
    ax.scatter(
        minimap_pts[:, 0], minimap_pts[:, 1], s=1, alpha=0.8, c='k', label="Mini‑map (merged)"
    )
    if query_pts_init is not None:
        ax.scatter(
            query_pts_init[:, 0],
            query_pts_init[:, 1],
            s=1,
            alpha=0.5,
            c='r',
            label="Query (init)",
        )
    if query_pts_refined is not None:
        ax.scatter(
            query_pts_refined[:, 0],
            query_pts_refined[:, 1],
            s=1,
            alpha=0.3,
            c='b',
            label="Query (refined)",
        )

    ax.set_aspect("equal")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, linestyle="--", alpha=0.3)
    ax.set_title("Query Alignment on Mini‑map", pad=12)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.legend(loc="upper right")
    if info_text:
        ax.text(
            0.01,
            0.99,
            info_text,
            transform=ax.transAxes,
            va="top",
            ha="left",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.9),
        )
    plt.tight_layout()
    plt.show()
