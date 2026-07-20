import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D

from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import open3d as o3d
import os
import glob

from tf_math import Transform

def plot_the_map_2d(points, traj, n_arrows=15, stride=None, xlim=None, ylim=None, map_name="keyframe_map"):
    x, y = traj[:,0], traj[:,1]
    fig, ax = plt.subplots(figsize=(10,10))

    ax.scatter(points[:,0], points[:,1], s=0.5, c='gray', alpha=0.5, label='Map')

    # Trajectory with green -> red gradient (start -> end), but with reversed colormap
    if len(x) > 1:
        xy   = np.column_stack([x, y])
        segs = np.stack([xy[:-1], xy[1:]], axis=1)
        t    = np.linspace(0.0, 1.0, len(segs))
        lc   = LineCollection(segs, cmap='brg_r', norm=Normalize(0,1))  # use reversed colormap
        lc.set_array(t)
        lc.set_linewidth(1.6)
        lc.set_alpha(0.9)
        ax.add_collection(lc)

    # Arrows + white index labels
    if (stride is not None and stride >= 1) or (n_arrows > 0):
        offset = 30
        if stride is not None and stride >= 1:
            idx = np.arange(offset, max(offset, len(x)-2-offset), stride, dtype=int)
        else:
            idx = np.linspace(offset, len(x)-2-offset, n_arrows, dtype=int)

        if idx.size > 0:
            dx, dy = x[idx+1]-x[idx], y[idx+1]-y[idx]
            nrm    = np.hypot(dx, dy)/0.1 + 1e-9
            ux, uy = dx/nrm, dy/nrm

            ax.quiver(x[idx], y[idx], ux, uy, alpha=0.7,
                      angles='xy', scale_units='xy', scale=0.1,
                      width=0.006, headwidth=5, color='black', zorder=3)

            # Perpendicular offset keeps labels readable on top of arrows
            angles_deg = np.degrees(np.arctan2(uy, ux))
            tx = x[idx] - uy*6
            ty = y[idx] + ux*6
            for i, xi, yi, ang in zip(idx, tx, ty, angles_deg):
                ax.text(xi, yi, str(i), color='white', fontsize=8,
                        ha='center', va='center', rotation=ang, rotation_mode='anchor', zorder=4,
                        path_effects=[pe.withStroke(linewidth=2.5, foreground='black')])

    ax.scatter(x[0], y[0], c='green', s=80, alpha=0.8, label='Start',   zorder=4)
    ax.scatter(x[-1], y[-1], c='red', s=80, alpha=0.8, label='Finish', zorder=4)
    ax.text(x[0],  y[0],  'Start',  fontsize=12, color='darkgreen', va='bottom', ha='right')
    ax.text(x[-1], y[-1], 'Finish', fontsize=12, color='darkred',   va='top',    ha='left')

    if xlim: ax.set_xlim(*xlim)
    if ylim: ax.set_ylim(*ylim)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_title(f'{map_name}, 2D Trajectory')

    # Legend (add a proxy for the gradient trajectory)
    handles, labels = ax.get_legend_handles_labels()
    traj_proxy = Line2D([0],[0], lw=2, color='black', label='Trajectory (green→red→blue)')
    handles.insert(1, traj_proxy)  # place near the top
    labels.insert(1, 'Trajectory (green→red)')
    ax.legend(handles, labels, loc='upper left')

    ax.grid(True); plt.tight_layout(); plt.show()
    
# Trajectory
def load_trajectory_df(traj_path, sep=","):
    traj_df = pd.read_table(
        traj_path, header=None, sep=sep,
        names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
        comment='#'
    )
    print(f"Loaded frames with {len(traj_df)} entries.")
    return traj_df

# Lidar
def load_keyframemap(lidar_dir: Path, traj_df:pd.DataFrame):
    pcd_dict = {}
    for pcd_path in sorted(glob.glob(os.path.join(lidar_dir, "*.pcd"))):
        idx = int(os.path.basename(pcd_path).split('.')[0])
        t, x, y, z, qx, qy, qz, qw = traj_df.iloc[idx].values
        T_lidar_TO_map = Transform.from_quat([qx, qy, qz, qw], [x, y, z]).as_transformation_matrix()
        pc = o3d.io.read_point_cloud(pcd_path)
        pc.transform(T_lidar_TO_map)
        pcd_dict[idx] = pc
    return pcd_dict

def merge_full(pcd_dict):
    m = o3d.geometry.PointCloud()
    for scan in pcd_dict.values():
        m += scan
    return m

def load_map(lidar_dir: Path, traj_df:pd.DataFrame, voxel_size: float = 0.3) -> np.ndarray:
    map = merge_full(load_keyframemap(lidar_dir, traj_df))
    map = map.voxel_down_sample(voxel_size)
    points = np.asarray(map.points)
    print(f"Loaded map with {points.shape[0]} points.")
    return points
