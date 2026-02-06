#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from scipy.spatial.transform import Rotation
import open3d as o3d

from opr.inference.index import FaissFlatIndex
from opr.inference.pipelines import PlaceRecognitionPipeline
from opr.inference.preprocessing import PointCloudMinkPreprocessor
from opr.models.place_recognition import MinkLoc3D
from opr.utils import parse_device

from mmpr.data.transforms import get_T_map_to_world
from mmpr.pr_cache import PerFramePR, save_pr_cache_npz


# == Computed transformation to the map1' ==
T_2_1 = np.array([
    [ 0.956711209165,  0.290813077207, -0.011463698546, -0.153455820707],
    [-0.290812256493,  0.956778490454,  0.001775296591,  0.688947242214],
    [ 0.011484499655,  0.001635337894,  0.999932713705, -0.077340220372],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_3_1 = np.array([
    [ 0.994962719425, -0.100235832042, -0.001401759667,  0.514291026771],
    [ 0.100237186633,  0.994963134041,  0.000931834714,  0.461334808929],
    [ 0.001301295964, -0.001067649246,  0.999998583376, -0.130197717405],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_4_1 = np.array([
    [ 0.802577217171, -0.595935887259, -0.027022745116,  0.386908564150],
    [ 0.595360363748,  0.803014107814, -0.026727886735,  0.886000197368],
    [ 0.037627752456,  0.005362921595,  0.999277434608, -0.097609346688],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_5_1 = np.array([
    [-0.314015910006, -0.949409051338, -0.004057277569,  18.696274552193],
    [ 0.949408164245, -0.313990708099, -0.005828627066,  11.209595424855],
    [ 0.004259803837, -0.005682294081,  0.999974782485, -0.462489823486],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_6_1 = np.array([
    [-0.666876352700, -0.745157342238,  0.004057772695,  18.889920284467],
    [ 0.745168270604, -0.666869447050,  0.003064159890,  11.280045723329],
    [ 0.000422723393,  0.005067139233,  0.999987072619, -0.427276360094],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_7_1 = np.array([
    [-0.608371581503, -0.793649920732, -0.001955029598,  18.883837684948],
    [ 0.793644427264, -0.608374670851,  0.002963602522,  11.397353413000],
    [-0.003541453394,  0.000251373208,  0.999993697440, -0.461398363017],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)
T_8_1 = np.array([
    [-0.989549287520, -0.144184982476,  0.001702467792,  16.858101263295],
    [ 0.144180168093, -0.989547836670, -0.002675456918, -11.745636334855],
    [ 0.002070434030, -0.002402034395,  0.999994971754, -0.467016966506],
    [ 0.000000000000,  0.000000000000,  0.000000000000,  1.000000000000],
], dtype=np.float32)


def mat44_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 to [tx,ty,tz,qx,qy,qz,qw]."""
    Rm = T[:3, :3]
    t = T[:3, 3]
    q = Rotation.from_matrix(Rm).as_quat()
    return np.concatenate([t, q]).astype(np.float32)


def load_pcd_points(path: Path, voxel_size: float | None = None) -> np.ndarray:
    pc = o3d.io.read_point_cloud(str(path))
    if voxel_size is not None and voxel_size > 0.0:
        pc = pc.voxel_down_sample(float(voxel_size))
    pts = np.asarray(pc.points)
    if pts.size == 0:
        return pts.reshape(0, 3).astype(np.float32)
    return pts.astype(np.float32)


class SimplePCDLoader:
    """Load keyframe scans and poses from `<map_dir>/keyframe_map`.

    This mirrors the notebook loader and supports an optional transform to map1/world.
    """

    def __init__(self, map_root: Path, scans_subdir: str = "scans", T_map_to_world: Optional[np.ndarray] = None) -> None:
        self._map_root = map_root
        self._scans_subdir = scans_subdir
        traj_path = self._map_root / "poses.csv"
        self._poses = self._load_poses(traj_path)  # [N,7]

        if T_map_to_world is not None:
            # Transform poses: T_w<-i = T_w<-map * T_map<-i
            num = self._poses.shape[0]
            out = np.zeros_like(self._poses, dtype=np.float32)
            for i in range(num):
                pose7 = self._poses[i]
                t = pose7[:3]
                q = pose7[3:]
                Rm = Rotation.from_quat(q).as_matrix().astype(np.float32)
                T = np.eye(4, dtype=np.float32)
                T[:3, :3] = Rm
                T[:3, 3] = t
                Twi = T_map_to_world @ T
                out[i] = mat44_to_pose7(Twi)
            self._poses = out

        self._scan_paths = sorted(list((self._map_root / self._scans_subdir).glob("*.pcd")))
        if len(self._scan_paths) != len(self._poses):
            raise RuntimeError(
                f"#scans ({len(self._scan_paths)}) != #poses ({len(self._poses)}) in {self._map_root}"
            )

    def __len__(self) -> int:
        return len(self._scan_paths)

    def __getitem__(self, idx: int) -> tuple[np.ndarray, np.ndarray, Path]:
        scan_path = self._scan_paths[idx]
        pose7 = self._poses[idx]
        points = load_pcd_points(scan_path)
        return points, pose7, scan_path

    @staticmethod
    def _load_poses(traj_path: Path) -> np.ndarray:
        import pandas as pd

        df = pd.read_table(
            traj_path,
            header=None,
            sep=",",
            names=["timestamp", "x", "y", "z", "qx", "qy", "qz", "qw"],
            comment="#",
        )
        vals = df[["x", "y", "z", "qx", "qy", "qz", "qw"]].to_numpy(dtype=np.float32)
        return vals


def get_T_map_to_world(map_name: str) -> np.ndarray:
    name = str(map_name).strip().lower()
    if name == "map1":
        return np.eye(4, dtype=np.float32)
    mapping: dict[str, np.ndarray] = {
        "map2": T_2_1,
        "map3": T_3_1,
        "map4": T_4_1,
        "map5": T_5_1,
        "map6": T_6_1,
        "map7": T_7_1,
        "map8": T_8_1,
    }
    if name in mapping:
        return mapping[name]
    print(f"[WARN] Unknown map name '{map_name}', using identity transform to map1/world")
    return np.eye(4, dtype=np.float32)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run PlaceRecognitionPipeline and save per-frame PR caches (npz) per map")
    p.add_argument("--root-data-dir", type=Path, required=True)
    p.add_argument("--map", dest="map_name", type=str, required=True)
    p.add_argument("--db-map-dir", type=Path, required=True, help="Path to DB keyframe_map (map1)")
    p.add_argument("--index-dir", type=Path, required=True, help="Directory with FAISS index for DB")
    p.add_argument("--weights", type=Path, required=True, help="Model weights path for MinkLoc3D")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--per-frame-k", type=int, default=100)
    p.add_argument("--pr-quant-size", type=float, default=0.05)
    p.add_argument("--output", type=Path, required=True, help="Output npz file per map")
    return p.parse_args()


def main() -> None:
    a = parse_args()
    device = parse_device(a.device)

    # Load index and model
    index = FaissFlatIndex.load(str(a.index_dir))
    model = MinkLoc3D()
    pr = PlaceRecognitionPipeline(index=index, model=model, model_weights_path=str(a.weights), device=device)

    # Preprocessor
    pre = PointCloudMinkPreprocessor(quantization_size=float(a.pr_quant_size), use_intensity=False)

    map_dir = (a.root_data_dir / a.map_name / "keyframe_map").resolve()
    T_map_to_world = get_T_map_to_world(a.map_name)
    loader = SimplePCDLoader(map_dir, T_map_to_world=T_map_to_world)

    frames: list[PerFramePR] = []
    for idx in range(len(loader)):
        points, _pose7, _path = loader[idx]
        pr_input: dict[str, Tensor] = pre(points)
        pr_input = {k: v.to(device) for k, v in pr_input.items()}
        res = pr.infer(pr_input, k=int(a.per_frame_k))
        frames.append(PerFramePR(indices=res.indices, distances=res.distances, db_idx=res.db_idx))

    a.output.parent.mkdir(parents=True, exist_ok=True)
    save_pr_cache_npz(a.output, frames)
    print(f"Saved PR cache: {a.output}")


if __name__ == "__main__":
    main()


