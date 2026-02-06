#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import numpy as np
import open3d as o3d
import rerun as rr
from scipy.spatial.transform import Rotation

# OPR helpers
from opr.inference.io import PointCloudStore, load_localization_results_jsonl


# -------------------------------
# Colors (fixed palette)
# -------------------------------
MAP_COLOR = [200, 200, 200]
QUERY_GT_COLOR = [255, 64, 64]
CAND_DB_COLOR = [64, 200, 255]
AXIS_X_COLOR = [255, 0, 0]
AXIS_Y_COLOR = [0, 255, 0]
AXIS_Z_COLOR = [0, 128, 255]

# Distinct rank colors (Matplotlib Tab10 palette in 0-255 RGB)
RANK_COLORS = [
    [31, 119, 180],   # 1
    [255, 127, 14],   # 2
    [44, 160, 44],    # 3
    [214, 39, 40],    # 4
    [148, 103, 189],  # 5
    [140, 86, 75],    # 6
    [227, 119, 194],  # 7
    [127, 127, 127],  # 8
    [188, 189, 34],   # 9
    [23, 190, 207],   # 10
]

def brighten_rgb(rgb: list[int] | np.ndarray, factor: float = 1.25) -> np.ndarray:
    c = np.array(rgb, dtype=np.float32) * float(factor)
    return np.clip(c, 0.0, 255.0).astype(np.uint8)


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



@dataclass
class Args:
    root_data_dir: Path
    map_name: str
    db_map_dir: Path
    results_jsonl: Path
    map1_voxelized_pcd: Path
    transform_path: Optional[Path]
    transforms_json: Optional[Path]
    top_n: int
    map_voxel_size: float
    voxel_size: float
    axes_scale: float
    point_radius: float
    max_frames: Optional[int]
    output: Path
    app_id: str


def parse_args() -> Args:
    p = argparse.ArgumentParser(description="Record Rerun .rrd visualization for OPR localization results (per map)")
    p.add_argument("--root-data-dir", type=Path, required=True, help="Path to dataset root containing mapX directories")
    p.add_argument("--map", dest="map_name", type=str, required=True, help="Map name to visualize (e.g., map2)")
    p.add_argument(
        "--db-map-dir",
        type=Path,
        default=None,
        help="Path to database map1 keyframe_map directory (default: <root-data-dir>/map1/keyframe_map)",
    )
    p.add_argument(
        "--results-jsonl",
        type=Path,
        required=True,
        help="Path to saved localization results JSONL for the selected map (from notebooks)",
    )
    p.add_argument(
        "--map1-voxelized-pcd",
        type=Path,
        default=None,
        help="Path to pre-voxelized map1 point cloud .pcd (default: <root>/map1/metric_map/map_voxelized_0.3.pcd)",
    )
    # Removed external transform loading; using built-in constants per user request
    p.add_argument("--top-n", type=int, default=3, help="Number of PR candidates to visualize per query")
    p.add_argument("--map-voxel", type=float, default=0.5, help="Voxel size for map pointcloud (applied on load)")
    p.add_argument("--voxel", type=float, default=0.1, help="Voxel size for query and candidate DB pointclouds")
    p.add_argument("--axes-scale", type=float, default=2.0, help="Length of per-pose axes arrows (meters)")
    p.add_argument("--point-radius", type=float, default=0.02, help="Radius for rendered points")
    p.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit the maximum number of frames logged (None = no limit)",
    )
    p.add_argument("--output", type=Path, required=True, help="Output .rrd path (separate per map)")
    p.add_argument("--app-id", type=str, default="opr_localization_viz", help="Rerun app id")

    a = p.parse_args()
    root_dir = a.root_data_dir.resolve()
    db_map_dir = a.db_map_dir if a.db_map_dir is not None else (root_dir / "map1" / "keyframe_map")
    map1_pcd = (
        a.map1_voxelized_pcd
        if a.map1_voxelized_pcd is not None
        else (root_dir / "map1" / "metric_map" / "map_voxelized_0.3.pcd")
    )

    return Args(
        root_data_dir=root_dir,
        map_name=str(a.map_name),
        db_map_dir=db_map_dir.resolve(),
        results_jsonl=a.results_jsonl.resolve(),
        map1_voxelized_pcd=map1_pcd.resolve(),
        transform_path=None,
        transforms_json=None,
        top_n=int(a.top_n),
        map_voxel_size=float(a.map_voxel),
        voxel_size=float(a.voxel),
        axes_scale=float(a.axes_scale),
        point_radius=float(a.point_radius),
        max_frames=(int(a.max_frames) if a.max_frames is not None else None),
        output=a.output.resolve(),
        app_id=str(a.app_id),
    )


def load_transform(path: Optional[Path]) -> np.ndarray:
    """Load 4x4 transform from file or return identity if None."""
    if path is None:
        return np.eye(4, dtype=np.float32)

    if path.suffix.lower() == ".npy":
        T = np.load(path)
    else:
        txt = Path(path).read_text().strip().split()
        vals = list(map(float, txt))
        if len(vals) == 16:
            T = np.array(vals, dtype=np.float32).reshape(4, 4)
        elif len(vals) == 7:
            t = np.array(vals[:3], dtype=np.float32)
            q = np.array(vals[3:], dtype=np.float32)
            Rm = Rotation.from_quat(q).as_matrix().astype(np.float32)
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = Rm
            T[:3, 3] = t
        else:
            raise ValueError(f"Unsupported transform format in {path}: expected 16 or 7 floats, got {len(vals)}")
    if T.shape != (4, 4):
        raise ValueError(f"Transform at {path} is not 4x4, got shape {T.shape}")
    return T.astype(np.float32)


def _to_mat4x4_from_json_value(val: Iterable[float]) -> np.ndarray:
    arr = np.array(list(val), dtype=np.float32).reshape(-1)
    if arr.size == 16:
        return arr.reshape(4, 4)
    if arr.size == 7:
        t = arr[:3]
        q = arr[3:]
        Rm = Rotation.from_quat(q).as_matrix().astype(np.float32)
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = Rm
        T[:3, 3] = t
        return T
    raise ValueError(f"Transform JSON value must have 16 or 7 floats; got {arr.size}")


def resolve_transform_for_map(
    map_name: str,
    root_data_dir: Path,
    map_dir: Path,
    override_path: Optional[Path],
    transforms_json: Optional[Path],
) -> np.ndarray:
    """Resolve T_map_to_world for selected map.

    Priority:
      1) explicit file via --tf-map-to-map1
      2) mapping file via --transforms-json or <root>/transforms_to_map1.json
      3) per-map files inside map dir: transform_to_map1.(npy|txt|json) or T_<id>_1.(npy|txt|json)
      4) identity for map1; otherwise identity with warning
    """
    if override_path is not None:
        return load_transform(override_path)

    if map_name == "map1":
        return np.eye(4, dtype=np.float32)

    try:
        json_path = transforms_json if transforms_json is not None else (root_data_dir / "transforms_to_map1.json")
        if json_path.exists():
            import json

            data = json.loads(json_path.read_text())
            if map_name in data:
                return _to_mat4x4_from_json_value(data[map_name])
    except Exception:
        pass

    candidates: list[Path] = []
    candidates.append(map_dir / "transform_to_map1.npy")
    candidates.append(map_dir / "transform_to_map1.txt")
    candidates.append(map_dir / "transform_to_map1.json")
    try:
        map_id = map_name.lower().replace("map", "").strip()
        if map_id:
            candidates.append(map_dir / f"T_{map_id}_1.npy")
            candidates.append(map_dir / f"T_{map_id}_1.txt")
            candidates.append(map_dir / f"T_{map_id}_1.json")
    except Exception:
        pass

    for p in candidates:
        if p.exists():
            if p.suffix.lower() == ".json":
                import json

                arr = json.loads(p.read_text())
                return _to_mat4x4_from_json_value(arr)
            return load_transform(p)

    print(f"[WARN] Transform to map1 not found for {map_name}. Using identity.")
    return np.eye(4, dtype=np.float32)


def pose7_to_transform3d(pose7: np.ndarray) -> rr.Transform3D:
    """Convert [tx,ty,tz,qx,qy,qz,qw] to Rerun Transform3D."""
    t = pose7[:3].astype(np.float32)
    q = pose7[3:].astype(np.float32)
    return rr.Transform3D(translation=t.tolist(), rotation=rr.Quaternion(xyzw=q.tolist()))


def mat44_to_pose7(T: np.ndarray) -> np.ndarray:
    """Convert 4x4 to [tx,ty,tz,qx,qy,qz,qw]."""
    Rm = T[:3, :3]
    t = T[:3, 3]
    q = Rotation.from_matrix(Rm).as_quat()
    return np.concatenate([t, q]).astype(np.float32)


def load_pcd_points(path: Path, voxel_size: Optional[float] = None) -> np.ndarray:
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


def log_axes_at(path: str, axes_scale: float) -> None:
    """Log three small arrows as XYZ axes below an already-logged Transform3D entity."""
    vectors = np.array(
        [
            [axes_scale, 0.0, 0.0],
            [0.0, axes_scale, 0.0],
            [0.0, 0.0, axes_scale],
        ],
        dtype=np.float32,
    )
    origins = np.zeros_like(vectors)
    colors = np.array([AXIS_X_COLOR, AXIS_Y_COLOR, AXIS_Z_COLOR], dtype=np.uint8)
    rr.log(path, rr.Arrows3D(vectors=vectors, origins=origins, colors=colors, radii=max(axes_scale * 0.08, 0.04)))


def ensure_parent_dir(path: Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def log_rank_annotation_context(k: int) -> None:
    """Provide rank labels/colors via AnnotationContext for consistent styling and labeling."""
    k = int(max(0, min(k, 10)))
    try:
        infos = []
        for r in range(k):
            rr_id = r + 1
            color = RANK_COLORS[min(r, len(RANK_COLORS) - 1)]
            # Prefer AnnotationInfo; fall back silently if not available in SDK
            try:
                info = rr.AnnotationInfo(id=rr_id, label=f"{rr_id}", color=tuple(color))
                infos.append(info)
            except Exception:
                pass
        if infos:
            rr.log("/", rr.AnnotationContext(infos), static=True)
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    # Resolve directories
    map_dir = args.root_data_dir / args.map_name / "keyframe_map"
    if not map_dir.exists():
        raise FileNotFoundError(f"Map directory not found: {map_dir}")
    if not args.db_map_dir.exists():
        raise FileNotFoundError(f"DB map directory not found: {args.db_map_dir}")
    if not args.map1_voxelized_pcd.exists():
        raise FileNotFoundError(f"Map1 voxelized PCD not found: {args.map1_voxelized_pcd}")
    if not args.results_jsonl.exists():
        raise FileNotFoundError(f"Results JSONL not found: {args.results_jsonl}")

    # Prepare loaders
    T_map_to_world = get_T_map_to_world(args.map_name)
    query_loader = SimplePCDLoader(map_dir, T_map_to_world=T_map_to_world)
    db_store = PointCloudStore(root_dir=args.db_map_dir)

    # Rerun recording setup
    ensure_parent_dir(args.output)
    rr.init(f"{args.app_id}_{args.map_name}")
    rr.save(str(args.output))

    # Log static world map (already voxelized)
    map_points = load_pcd_points(args.map1_voxelized_pcd, voxel_size=args.map_voxel_size)
    rr.log("world/map1", rr.Transform3D())
    if map_points.size:
        rr.log("world/map1/points", rr.Points3D(map_points, colors=MAP_COLOR, radii=args.point_radius), static=True)
    # (By request) Do not show previous queries at a given timestamp; skip static trajectory logging

    # Load predictions
    predictions = load_localization_results_jsonl(str(args.results_jsonl))

    # Cache for DB clouds (relative path -> ndarray)
    db_cache: dict[str, np.ndarray] = {}

    # Per-query logging
    for idx, pred in enumerate(predictions):
        if args.max_frames is not None and idx >= int(args.max_frames):
            break
        if idx >= len(query_loader):
            break
        # Timeline step
        try:
            rr.set_time("frame", sequence=idx)
        except Exception:
            pass

        # Load query scan and GT pose
        query_pts, query_pose7, scan_path = query_loader[idx]

        # Log GT pose transform + cloud (constant path -> only current frame shows)
        rr.log("world/query/gt", pose7_to_transform3d(query_pose7))
        if query_pts.size:
            q_pts_ds = load_pcd_points(scan_path, voxel_size=args.voxel_size)
            rr.log(
                "world/query/gt/points",
                rr.Points3D(q_pts_ds, colors=QUERY_GT_COLOR, radii=args.point_radius),
            )
        else:
            rr.log("world/query/gt/points", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))

        # Candidates: show only estimated poses (top-10 or top-N if smaller) and draw lines from GT to est
        max_allowed = 10
        max_k = min(int(args.top_n), max_allowed, len(pred.candidates))

        # Precompute GT origin in world to anchor lines
        gt_T = np.eye(4, dtype=np.float32)
        gt_q = np.array(query_pose7[3:], dtype=np.float32)
        gt_t = np.array(query_pose7[:3], dtype=np.float32)
        gt_T[:3, :3] = Rotation.from_quat(gt_q).as_matrix().astype(np.float32)
        gt_T[:3, 3] = gt_t
        gt_origin = gt_t

        # Log GT axes (default colors) at a constant path and add "Q" label
        rr.log("world/query/gt_pose", pose7_to_transform3d(query_pose7))
        q_scale = args.axes_scale
        q_vec = np.array([[q_scale, 0.0, 0.0], [0.0, q_scale, 0.0], [0.0, 0.0, q_scale]], dtype=np.float32)
        rr.log("world/query/gt_pose/axes", rr.Arrows3D(vectors=q_vec, origins=np.zeros_like(q_vec), radii=max(q_scale * 0.08, 0.04)))
        try:
            rr.log("world/query/gt_pose", rr.AnnotationContext([rr.AnnotationInfo(id=0, label="Q", color=(255, 255, 255))]), static=True)
        except Exception:
            pass

        # Populate current top-K estimated poses with rank-based colors, and connect GT→DB and DB→EST
        for rank in range(max_k):
            cand = pred.candidates[rank]
            color = np.array(RANK_COLORS[min(rank, len(RANK_COLORS) - 1)], dtype=np.uint8)
            group_root = f"world/estimates/{rank+1:02d}"

            # Estimated pose axes (full scale)
            rr.log(group_root + "/pose", pose7_to_transform3d(np.array(cand.estimated_pose, dtype=np.float32)))
            # Use rank color for the axes arrows (by drawing our own colored arrows)
            # Draw axes with default RGB (no rank colorization)
            scale = args.axes_scale
            vectors = np.array([[scale, 0.0, 0.0], [0.0, scale, 0.0], [0.0, 0.0, scale]], dtype=np.float32)
            origins = np.zeros_like(vectors)
            rr.log(group_root + "/pose/axes", rr.Arrows3D(vectors=vectors, origins=origins, radii=max(scale * 0.08, 0.04)))
            # Per-pose AnnotationContext at estimated pose center with label "i'" (colorized)
            try:
                est_info = rr.AnnotationInfo(id=rank + 1, label=f"{rank + 1}'", color=tuple(RANK_COLORS[min(rank, len(RANK_COLORS) - 1)]))
                rr.log(group_root + "/pose", rr.AnnotationContext([est_info]), static=True)
            except Exception:
                pass

            # DB pose (half-sized axes), per-pose AnnotationContext, and DB cloud under that transform
            rr.log(group_root + "/db_pose", pose7_to_transform3d(np.array(cand.db_pose, dtype=np.float32)))
            # DB axes same size as estimated: use args.axes_scale
            db_scale = max(args.axes_scale, 0.01)
            vectors = np.array([[db_scale, 0.0, 0.0], [0.0, db_scale, 0.0], [0.0, 0.0, db_scale]], dtype=np.float32)
            origins = np.zeros_like(vectors)
            rr.log(group_root + "/db_pose/axes", rr.Arrows3D(vectors=vectors, origins=origins, radii=max(db_scale * 0.08, 0.04)))

            # Per-pose AnnotationContext at DB pose center with label "i" (colorized)
            try:
                info = rr.AnnotationInfo(id=rank + 1, label=f"{rank + 1}", color=tuple(color.tolist()))
                rr.log(group_root + "/db_pose", rr.AnnotationContext([info]), static=True)
            except Exception:
                pass

            # DB cloud colored by rank (using class_id if supported), transformed via db_pose parent
            rel_path = cand.db_pointcloud_path
            pts = np.zeros((0, 3), dtype=np.float32)
            if isinstance(rel_path, str) and rel_path:
                if rel_path not in db_cache:
                    try:
                        db_pts = db_store.load(rel_path)
                        if isinstance(db_pts, np.ndarray):
                            pass
                        else:
                            db_pts = db_pts.cpu().numpy() if hasattr(db_pts, "cpu") else np.asarray(db_pts)
                        pc = o3d.geometry.PointCloud()
                        pc.points = o3d.utility.Vector3dVector(db_pts.astype(np.float32))
                        pc = pc.voxel_down_sample(float(args.voxel_size))
                        db_cache[rel_path] = np.asarray(pc.points).astype(np.float32)
                    except Exception:
                        db_cache[rel_path] = np.zeros((0, 3), dtype=np.float32)
                pts = db_cache.get(rel_path, np.zeros((0, 3), dtype=np.float32))
            try:
                rr.log(group_root + "/db_pose/points", rr.Points3D(pts, class_ids=np.full((pts.shape[0],), rank + 1, dtype=np.uint16), radii=args.point_radius))
            except Exception:
                rr.log(group_root + "/db_pose/points", rr.Points3D(pts, colors=color.tolist(), radii=args.point_radius))

            # Lines: GT→DB and DB→EST
            db_T = np.eye(4, dtype=np.float32)
            db_pose = np.array(cand.db_pose, dtype=np.float32)
            db_T[:3, :3] = Rotation.from_quat(db_pose[3:]).as_matrix().astype(np.float32)
            db_T[:3, 3] = db_pose[:3]
            db_origin = db_T[:3, 3]

            est_T = np.eye(4, dtype=np.float32)
            est_pose = np.array(cand.estimated_pose, dtype=np.float32)
            est_T[:3, :3] = Rotation.from_quat(est_pose[3:]).as_matrix().astype(np.float32)
            est_T[:3, 3] = est_pose[:3]
            est_origin = est_T[:3, 3]

            try:
                rr.log(
                    group_root + "/line_q_to_db",
                    rr.LineStrips3D(
                        np.stack([gt_origin, db_origin], axis=0).reshape(1, 2, 3),
                        class_ids=np.array([rank + 1], dtype=np.uint16),
                        radii=max(args.point_radius * 1.5, 0.03),
                    ),
                )
                rr.log(
                    group_root + "/line_db_to_est",
                    rr.LineStrips3D(
                        np.stack([db_origin, est_origin], axis=0).reshape(1, 2, 3),
                        class_ids=np.array([rank + 1], dtype=np.uint16),
                        radii=max(args.point_radius * 1.5, 0.03),
                    ),
                )
            except Exception:
                rr.log(
                    group_root + "/line_q_to_db",
                    rr.LineStrips3D(
                        np.stack([gt_origin, db_origin], axis=0).reshape(1, 2, 3),
                        colors=brighten_rgb(color).tolist(),
                        radii=max(args.point_radius * 1.5, 0.03),
                    ),
                )
                rr.log(
                    group_root + "/line_db_to_est",
                    rr.LineStrips3D(
                        np.stack([db_origin, est_origin], axis=0).reshape(1, 2, 3),
                        colors=brighten_rgb(color).tolist(),
                        radii=max(args.point_radius * 1.5, 0.03),
                    ),
                )

        # Clear any leftover estimate slots beyond max_k up to min(top_n, 10)
        empty_line = rr.LineStrips3D(np.zeros((1, 0, 3), dtype=np.float32))
        for rank in range(max_k, min(int(args.top_n), 10)):
            group_root = f"world/estimates/{rank+1:02d}"
            rr.log(group_root + "/pose/axes", rr.Arrows3D(vectors=np.zeros((0, 3), dtype=np.float32), origins=np.zeros((0, 3), dtype=np.float32)))
            rr.log(group_root + "/line_q_to_db", empty_line)
            rr.log(group_root + "/line_db_to_est", empty_line)
            rr.log(group_root + "/db_pose/points", rr.Points3D(np.zeros((0, 3), dtype=np.float32)))

    # Log legend once (static)
    # Prefer AnnotationContext-based legend; fall back to no legend if unsupported
    log_rank_annotation_context(k=min(int(args.top_n), 10))

    print(f"Saved Rerun recording to {args.output}")


if __name__ == "__main__":
    main()


