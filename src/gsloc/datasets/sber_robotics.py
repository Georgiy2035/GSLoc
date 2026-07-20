"""SberRobotics benchmark dataset loader and metadata builder.

Dataset layout (under ``/mnt/external_usb_hdd/6YL/Datasets/SberRobotics``):

``maps/<map_id>/keyframe_map/keyframe_map/poses.csv``
``maps/<map_id>/keyframe_map/keyframe_map/zedxone_left/rgb/000000.jpg``
``maps/SceneGraphs_pt/<map_id>/000000.pt``

Poses and asset naming follow ``mmpr.data.mmpr_data_reader.MmprFramesDataReader``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Union

import cv2
import numpy as np
import pandas as pd
import torch
from loguru import logger
from scipy.spatial.transform import Rotation as R
from torch import Tensor

from gsloc.datasets.pr_dataset import PRDataset
from gsloc.models.graph_encoder import EdgeAttrNormalizer
from gsloc.utils.graphs import (
    _collate_graph_objects,
    _ensure_nonempty,
    _sanitize_graph_obj,
    rotate_graph_features,
)
from mmpr.data.transforms import get_T_map_to_world
from mmpr.modules.vis_utils import quaternion_angle
from opr.datasets.augmentations import DefaultImageTransform

DEFAULT_DATASET_ROOT = Path("/mnt/external_usb_hdd/6YL/Datasets/SberRobotics")
DEFAULT_GRAPH_REL_PATH = Path("maps/SceneGraphs_pt")
DEFAULT_KEYFRAME_REL_PATH = Path("keyframe_map/keyframe_map")
DEFAULT_ROOM_ID = "office"

# Logical camera names used in mmpr loaders -> on-disk folder layout.
CAMERA_DIR_MAP: Dict[str, str] = {
    "cam_pinhole_left": "zedx_front_left/rgb",
    "cam_pinhole_right": "zedx_front_right/rgb",
    "cam_fish-eye_left": "zedxone_left/rgb",
    "cam_fish-eye-right": "zedxone_right/rgb",
}


def _load_scene_ids(scene_list_path: Union[str, Path]) -> Set[str]:
    scene_list_path = Path(scene_list_path)
    if not scene_list_path.exists():
        raise FileNotFoundError(f"Missing scene list file: {scene_list_path}")
    return {
        line.strip()
        for line in scene_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build_map_to_room_map(
    room_json_path: Optional[Union[str, Path]] = None,
    *,
    default_room: str = DEFAULT_ROOM_ID,
) -> Dict[str, str]:
    """Build ``map_id -> room_id`` mapping.

    If ``room_json_path`` is provided, it must be a JSON list of objects with
    ``reference`` (room id) and ``maps`` (list of map ids), similar to 3RScan.json.
    Otherwise every discovered map is assigned ``default_room``.
    """
    if room_json_path is None:
        return {}

    room_json_path = Path(room_json_path)
    if not room_json_path.exists():
        raise FileNotFoundError(f"Missing SberRobotics room file: {room_json_path}")

    with room_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at top-level in {room_json_path}")

    map_to_room: Dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        room_ref = entry.get("reference")
        maps = entry.get("maps")
        if room_ref is None or not isinstance(maps, list):
            continue
        for map_id in maps:
            map_to_room[str(map_id)] = str(room_ref)

    if not map_to_room:
        raise RuntimeError(f"Failed to build map->room mapping from {room_json_path}")
    return map_to_room


def resolve_sber_robotics_scene_filter(
    *,
    scene_list_path: Optional[Union[str, Path]] = None,
    scene_filter_mode: str = "all",
    room_json_path: Optional[Union[str, Path]] = None,
    maps_dir: Union[str, Path] = "maps",
    dataset_root: Optional[Union[str, Path]] = None,
) -> Optional[Set[str]]:
    """Resolve map ids to keep (same modes as ThreeRScan)."""
    if scene_filter_mode == "all":
        return None

    if scene_list_path is None:
        raise ValueError("`scene_list_path` must be provided when `scene_filter_mode` is not 'all'")

    listed_map_ids = _load_scene_ids(scene_list_path)
    if scene_filter_mode == "listed":
        return listed_map_ids

    if scene_filter_mode == "same_room_excluding_listed":
        map_to_room = build_map_to_room_map(room_json_path)
        if not map_to_room and dataset_root is not None:
            root = Path(dataset_root)
            maps_root = root / maps_dir
            if maps_root.exists():
                map_to_room = {
                    p.name: DEFAULT_ROOM_ID
                    for p in sorted(maps_root.iterdir())
                    if p.is_dir() and p.name.startswith("map")
                }

        listed_rooms = {map_to_room[mid] for mid in listed_map_ids if mid in map_to_room}
        if not listed_rooms and map_to_room:
            listed_rooms = {DEFAULT_ROOM_ID}

        if not listed_rooms:
            raise RuntimeError(
                f"None of the listed maps from {scene_list_path} were found in room mapping"
            )

        return {
            map_id
            for map_id, room_id in map_to_room.items()
            if room_id in listed_rooms and map_id not in listed_map_ids
        }

    raise ValueError(
        "`scene_filter_mode` must be one of: 'all', 'listed', 'same_room_excluding_listed'"
    )


def _read_poses_csv(poses_path: Path) -> pd.DataFrame:
    """Read keyframe ``poses.csv`` (comment header + ts, px..qw columns)."""
    return pd.read_csv(
        poses_path,
        comment="#",
        header=None,
        names=["ts", "px", "py", "pz", "qx", "qy", "qz", "qw"],
    )


def _pose7_from_row(
    row: pd.Series,
    *,
    transform_to_map1: bool = False,
    map_id: str = "",
    transform_path: Optional[Path] = None,
) -> List[float]:
    """Convert a poses.csv row to [tx, ty, tz, qx, qy, qz, qw]."""
    t = np.array([row["px"], row["py"], row["pz"]], dtype=np.float64)
    q = np.array([row["qx"], row["qy"], row["qz"], row["qw"]], dtype=np.float64)
    Rm = R.from_quat(q).as_matrix().astype(np.float64)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = Rm
    T[:3, 3] = t

    if transform_to_map1:
        if transform_path is not None and transform_path.is_file():
            T_map1 = np.load(str(transform_path)).astype(np.float64)
        else:
            T_map1 = get_T_map_to_world(map_id).astype(np.float64)
        T = T_map1 @ T

    q_out = R.from_matrix(T[:3, :3]).as_quat()
    return [
        float(T[0, 3]),
        float(T[1, 3]),
        float(T[2, 3]),
        float(q_out[0]),
        float(q_out[1]),
        float(q_out[2]),
        float(q_out[3]),
    ]


def _camera_path_from_index(
    map_root: Path,
    camera: str,
    frame_idx: int,
) -> Path:
    if camera not in CAMERA_DIR_MAP:
        raise ValueError(
            f"Unknown camera {camera!r}; expected one of {sorted(CAMERA_DIR_MAP)}"
        )
    base = map_root / CAMERA_DIR_MAP[camera]
    stem = f"{frame_idx:06d}"
    for ext in ("png", "jpg", "jpeg"):
        p = base / f"{stem}.{ext}"
        if p.exists():
            return p
    return base / f"{stem}.png"


def iter_sber_robotics_frames(
    dataset_root: Union[str, Path],
    *,
    maps_dir: Union[str, Path] = "maps",
    keyframe_relpath: Union[str, Path] = DEFAULT_KEYFRAME_REL_PATH,
    modality: list[str] = ["image", "graph"],
    graph_path: Optional[Union[str, Path]] = None,
    camera: str = "cam_fish-eye_left",
    transform_poses_to_map1: bool = False,
) -> Iterable[Dict[str, Any]]:
    """Yield per-frame records for each map under ``maps/``."""
    root = Path(dataset_root)
    maps_root = root / maps_dir
    if not maps_root.exists():
        raise FileNotFoundError(f"Expected maps folder at {maps_root}")

    graph_root = root / (graph_path or DEFAULT_GRAPH_REL_PATH)
    keyframe_relpath = Path(keyframe_relpath)

    for map_dir in sorted(p for p in maps_root.iterdir() if p.is_dir() and p.name.startswith("map")):
        map_id = map_dir.name
        map_root = map_dir / keyframe_relpath
        poses_path = map_root / "poses.csv"
        if not map_root.exists() or not poses_path.exists():
            logger.warning("Skipping map {} (missing {} or poses.csv)", map_id, map_root)
            continue

        transform_path = map_root / "transform_to_map1.npy"
        poses_df = _read_poses_csv(poses_path)

        for frame_idx, (_, row) in enumerate(poses_df.iterrows()):
            frame: Dict[str, Any] = {
                "scene_id": map_id,
                "frame_idx": frame_idx,
                "poses_path": poses_path,
            }

            if "image" in modality:
                image_path = _camera_path_from_index(map_root, camera, frame_idx)
                if not image_path.is_file():
                    continue
                frame["image_path"] = image_path

            if "graph" in modality:
                graph_file = graph_root / map_id / f"{frame_idx:06d}.pt"
                if not graph_file.is_file():
                    continue
                frame["graph_path"] = graph_file

            try:
                frame["pose7"] = _pose7_from_row(
                    row,
                    transform_to_map1=transform_poses_to_map1,
                    map_id=map_id,
                    transform_path=transform_path if transform_path.is_file() else None,
                )
            except Exception:
                logger.exception("Failed reading pose for {} frame {}", map_id, frame_idx)
                continue

            yield frame


def build_sber_robotics_df(
    dataset_root: Union[str, Path],
    *,
    limit: Optional[int] = None,
    log_every: int = 50_000,
    modality: list[str] = ["image", "graph"],
    scene_ids: Optional[Set[str]] = None,
    map_to_room_map: Optional[Dict[str, str]] = None,
    maps_dir: Union[str, Path] = "maps",
    keyframe_relpath: Union[str, Path] = DEFAULT_KEYFRAME_REL_PATH,
    graph_path: Optional[Union[str, Path]] = None,
    camera: str = "cam_fish-eye_left",
    transform_poses_to_map1: bool = False,
    similarity_filter_mode: str = "none",
    similarity_trans_tol_m: float = 3.0,
    similarity_rot_tol_deg: float = 60.0,
) -> pd.DataFrame:
    """Build metadata dataframe for SberRobotics maps."""
    root = Path(dataset_root)
    if map_to_room_map is None:
        map_to_room_map = {}

    rows: List[Dict[str, Any]] = []
    n = 0
    last_kept_by_scene: Dict[str, Dict[str, Any]] = {}
    skipped_similar = 0

    if similarity_filter_mode not in {"none", "pose", "room"}:
        raise ValueError("`similarity_filter_mode` must be one of: 'none', 'pose', 'room'")

    if scene_ids is None:
        logger.info("Scanning SberRobotics dataset...")
    else:
        logger.info("Scanning SberRobotics dataset for {} selected maps...", len(scene_ids))

    for frame in iter_sber_robotics_frames(
        root,
        maps_dir=maps_dir,
        keyframe_relpath=keyframe_relpath,
        modality=modality,
        graph_path=graph_path,
        camera=camera,
        transform_poses_to_map1=transform_poses_to_map1,
    ):
        if scene_ids is not None and frame["scene_id"] not in scene_ids:
            continue

        map_id = frame["scene_id"]
        row: Dict[str, Any] = {
            "idx": n,
            "scene": map_id,
            "room": map_to_room_map.get(map_id, DEFAULT_ROOM_ID),
            "pose": frame["pose7"],
            "frame_idx": frame["frame_idx"],
        }
        if "image" in modality:
            row["image_path"] = str(frame["image_path"])
        if "graph" in modality:
            row["graph_path"] = str(frame["graph_path"])

        if similarity_filter_mode != "none":
            prev = last_kept_by_scene.get(map_id)
            if prev is not None:
                room_a = prev.get("room", None)
                room_b = row.get("room", None)
                is_similar = False

                if room_a is not None and room_b is not None and room_a == room_b:
                    if similarity_filter_mode == "room":
                        is_similar = True
                    else:
                        pose_a = torch.as_tensor(prev["pose"], dtype=torch.float64)
                        pose_b = torch.as_tensor(row["pose"], dtype=torch.float64)
                        trans_diff = float((pose_a[:3] - pose_b[:3]).norm())
                        if trans_diff <= float(similarity_trans_tol_m):
                            rot_diff_deg = quaternion_angle(
                                pose_a[3:].detach().cpu().numpy(),
                                pose_b[3:].detach().cpu().numpy(),
                                degrees=True,
                                normalize=True,
                            )
                            is_similar = rot_diff_deg <= float(similarity_rot_tol_deg)

                if is_similar:
                    skipped_similar += 1
                    continue

            last_kept_by_scene[map_id] = {"room": row["room"], "pose": row["pose"]}

        rows.append(row)
        n += 1
        if log_every and (n % log_every == 0):
            logger.info("Scanned {:,} frames...", n)
        if limit is not None and n >= limit:
            break

    if not rows:
        if scene_ids is None:
            raise RuntimeError(f"No frames found under {root}/{maps_dir}/map*/{keyframe_relpath}")
        raise RuntimeError("No frames found for the requested map selection")

    df = pd.DataFrame(rows)
    logger.info("Scanned {} rows", n)
    if similarity_filter_mode != "none":
        logger.info(
            "Similarity filtering skipped {} sequential frames (mode={})",
            skipped_similar,
            similarity_filter_mode,
        )
    return df


class SberRobotics(PRDataset):
    """SberRobotics office benchmark with ThreeRScan-compatible PR interface."""

    _map_to_room_map: Optional[Dict[str, str]] = None

    def __init__(
        self,
        dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
        *,
        meta_path: Optional[Union[str, Path]] = None,
        meta_file: str = "meta.parquet",
        modality: list[str] = ["image", "graph"],
        rebuild_meta: bool = False,
        save_meta: bool = False,
        limit: Optional[int] = None,
        scene_list_path: Optional[Union[str, Path]] = None,
        scene_filter_mode: str = "all",
        room_json_path: Optional[Union[str, Path]] = None,
        image_transform: Any = DefaultImageTransform(resize=(320, 192), train=False),
        graph_feat_dim: int = 4,
        graph_edge_attr_dim: int = 7,
        graph_rotate: bool = True,
        edge_normalizer_path: Optional[Union[str, Path]] = None,
        graph_path: Optional[Union[str, Path]] = DEFAULT_GRAPH_REL_PATH,
        maps_dir: Union[str, Path] = "maps",
        keyframe_relpath: Union[str, Path] = DEFAULT_KEYFRAME_REL_PATH,
        camera: str = "cam_fish-eye_left",
        transform_poses_to_map1: bool = True,
        similarity_filter_mode: str = "none",
        similarity_trans_tol_m: float = 1.0,
        similarity_rot_tol_deg: float = 15.0,
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.modality = modality
        self.maps_dir = Path(maps_dir)
        self.keyframe_relpath = Path(keyframe_relpath)
        self.camera = camera
        self.transform_poses_to_map1 = transform_poses_to_map1
        self.graph_path = Path(graph_path) if graph_path is not None else DEFAULT_GRAPH_REL_PATH

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Given dataset_root={self.dataset_root} doesn't exist")

        self.meta_path = (
            self.dataset_root / meta_file if meta_path is None else Path(meta_path) / meta_file
        )
        if scene_list_path is None and scene_filter_mode != "all":
            scene_list_path = self.dataset_root / "files" / "test_maps.txt"
        self.scene_list_path = Path(scene_list_path) if scene_list_path is not None else None
        self.scene_filter_mode = scene_filter_mode
        self.room_json_path = Path(room_json_path) if room_json_path is not None else None
        self.similarity_filter_mode = similarity_filter_mode
        self.similarity_trans_tol_m = float(similarity_trans_tol_m)
        self.similarity_rot_tol_deg = float(similarity_rot_tol_deg)

        selected_scene_ids = resolve_sber_robotics_scene_filter(
            scene_list_path=self.scene_list_path,
            scene_filter_mode=self.scene_filter_mode,
            room_json_path=self.room_json_path,
            maps_dir=self.maps_dir,
            dataset_root=self.dataset_root,
        )

        can_use_cached_meta = self.meta_path.exists() and not rebuild_meta
        if can_use_cached_meta:
            self.df = pd.read_parquet(self.meta_path)
        else:
            logger.info(
                "Rebuilding metadata for SberRobotics dataset"
                if self.meta_path.exists()
                else "Metadata not found, rebuilding metadata for SberRobotics dataset"
            )
            self.df = build_sber_robotics_df(
                self.dataset_root,
                limit=limit,
                modality=modality,
                scene_ids=selected_scene_ids,
                map_to_room_map=self._get_map_to_room_map(room_json_path=self.room_json_path),
                maps_dir=self.maps_dir,
                keyframe_relpath=self.keyframe_relpath,
                graph_path=self.graph_path,
                camera=self.camera,
                transform_poses_to_map1=self.transform_poses_to_map1,
                similarity_filter_mode=self.similarity_filter_mode,
                similarity_trans_tol_m=self.similarity_trans_tol_m,
                similarity_rot_tol_deg=self.similarity_rot_tol_deg,
            )

        if selected_scene_ids is not None and can_use_cached_meta:
            self.df = self.df[self.df["scene"].astype(str).isin(selected_scene_ids)].reset_index(
                drop=True
            )
            self.df["idx"] = np.arange(len(self.df), dtype=np.int64)

        if limit is not None and can_use_cached_meta:
            self.df = self.df.iloc[:limit].reset_index(drop=True)
            self.df["idx"] = np.arange(len(self.df), dtype=np.int64)

        missing_cols = {"idx", "pose", "graph_path", "scene", "image_path", "room"} - set(self.df.columns)
        if "graph" not in self.modality:
            missing_cols -= {"graph_path"}
        if "image" not in self.modality:
            missing_cols -= {"image_path"}
        if missing_cols:
            raise ValueError(
                f"{self.meta_path} is missing columns: {sorted(missing_cols)}, {self.df.columns}"
            )

        self.image_transform = image_transform
        self.graph_feat_dim = graph_feat_dim
        self.graph_edge_attr_dim = graph_edge_attr_dim
        self.graph_rotate = graph_rotate
        self.edge_normalizer: Optional[EdgeAttrNormalizer] = None
        self._edge_norm_dim_mismatch_warned = False
        _norm_path = Path(edge_normalizer_path) if edge_normalizer_path is not None else None
        if _norm_path is not None:
            if _norm_path.is_file():
                try:
                    norm_ckpt = torch.load(_norm_path, map_location="cpu", weights_only=False)
                    self.edge_normalizer = EdgeAttrNormalizer(
                        log_indices=norm_ckpt.get("log_indices"),
                    )
                    self.edge_normalizer.mean = torch.as_tensor(
                        norm_ckpt["mean"], dtype=torch.float32
                    )
                    self.edge_normalizer.std = torch.as_tensor(
                        norm_ckpt["std"], dtype=torch.float32
                    )
                except Exception:
                    logger.exception(
                        "Failed to load edge normalizer from {}; continuing without it",
                        _norm_path,
                    )
                    self.edge_normalizer = None
            else:
                logger.warning(
                    "Edge normalizer checkpoint not found at {}; graph edge_attr will not be normalized",
                    _norm_path,
                )
        else:
            logger.warning(
                "Edge normalizer path is None; graph edge_attr will not be normalized",
            )

        self._missing_asset_warn_count = 0

        if save_meta:
            self.save_meta_parquet(self.meta_path.parent, meta_file)

    @staticmethod
    def _coalesce_storage_path(row: Any, key: str) -> Optional[Path]:
        if key not in row.index:
            return None
        v = row[key]
        if v is None:
            return None
        try:
            if pd.isna(v):
                return None
        except TypeError:
            pass
        s = str(v).strip()
        if not s or s.lower() == "nan":
            return None
        return Path(s)

    def _warn_missing_asset(self, kind: str, path: Any, exc: Optional[BaseException] = None) -> None:
        self._missing_asset_warn_count += 1
        if self._missing_asset_warn_count <= 25:
            if exc is not None:
                logger.warning(
                    "Missing or unreadable {} {}: {} — {}", kind, path, type(exc).__name__, exc
                )
            else:
                logger.warning("Missing or unreadable {} {}", kind, path)
        elif self._missing_asset_warn_count == 26:
            logger.warning(
                "Further missing image/graph file warnings suppressed (first 25 were logged)"
            )

    def _synthetic_image_tensor(self) -> Tensor:
        img = np.zeros((240, 320, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_transform is not None:
            img = self.image_transform(img)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.ascontiguousarray(img))
            if img.ndim == 3:
                img = img.permute(2, 0, 1)
            img = img.float() / 255.0
        return img

    def _placeholder_graph(self) -> Any:
        g = _ensure_nonempty(None, self.graph_feat_dim, self.graph_edge_attr_dim)
        if self.graph_rotate:
            g = rotate_graph_features(g)
        self._apply_edge_normalizer(g)
        return g

    def _load_image(self, image_path: Optional[Union[str, Path]]) -> Tensor:
        if image_path is None:
            self._warn_missing_asset("image", "(null path in meta)")
            return self._synthetic_image_tensor()
        p = Path(image_path)
        if not p.is_file():
            self._warn_missing_asset("image", p)
            return self._synthetic_image_tensor()
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if img is None:
            self._warn_missing_asset("image", p)
            return self._synthetic_image_tensor()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_transform is not None:
            img = self.image_transform(img)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.ascontiguousarray(img))
            if img.ndim == 3:
                img = img.permute(2, 0, 1)
            img = img.float() / 255.0
        return img

    def _load_graph(self, graph_path: Optional[Union[str, Path]]) -> Any:
        if graph_path is None:
            self._warn_missing_asset("graph", "(null path in meta)")
            return self._placeholder_graph()
        p = Path(graph_path)
        if not p.is_file():
            self._warn_missing_asset("graph", p)
            return self._placeholder_graph()
        try:
            graph = torch.load(p, map_location="cpu", weights_only=False)
        except Exception as exc:
            self._warn_missing_asset("graph", p, exc)
            return self._placeholder_graph()

        graph = _sanitize_graph_obj(
            graph,
            feat_dim=self.graph_feat_dim,
            feat_edge_attr_dim=self.graph_edge_attr_dim,
        )

        if isinstance(graph, list):
            out = []
            for g in graph:
                g = _ensure_nonempty(g, self.graph_feat_dim, self.graph_edge_attr_dim)
                if self.graph_rotate:
                    g = rotate_graph_features(g)
                self._apply_edge_normalizer(g)
                out.append(g)
            return out

        graph = _ensure_nonempty(graph, self.graph_feat_dim, self.graph_edge_attr_dim)
        if self.graph_rotate:
            graph = rotate_graph_features(graph)
        self._apply_edge_normalizer(graph)
        return graph

    def _apply_edge_normalizer(self, graph: Any) -> None:
        if self.edge_normalizer is None:
            return
        ea = getattr(graph, "edge_attr", None)
        if ea is None or ea.numel() == 0:
            return
        m = self.edge_normalizer.mean
        s = self.edge_normalizer.std
        if m is None or s is None:
            return
        m_n = torch.as_tensor(m).numel()
        if m_n != ea.shape[1]:
            if not self._edge_norm_dim_mismatch_warned:
                self._edge_norm_dim_mismatch_warned = True
                logger.warning(
                    "Skipping edge normalization: normalizer stats dim {} != edge_attr dim {}",
                    m_n,
                    ea.shape[1],
                )
            return
        try:
            graph.edge_attr = self.edge_normalizer.transform(ea)
        except Exception:
            logger.exception("Edge normalizer transform failed; leaving edge_attr unchanged")

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]
        scene = row["scene"]
        room = row["room"]
        pose = torch.tensor(np.asarray(row["pose"], dtype=np.float32), dtype=torch.float32)

        frame: Dict[str, Any] = {
            "idx": torch.tensor(int(row["idx"]), dtype=torch.int64),
            "scene": str(scene),
            "room": str(room),
            "scene_hash": torch.tensor(hash(str(scene))),
            "pose": pose,
        }
        if "image" in self.modality:
            frame["image_main"] = self._load_image(self._coalesce_storage_path(row, "image_path"))
        if "graph" in self.modality:
            frame["graph_main"] = self._load_graph(self._coalesce_storage_path(row, "graph_path"))
        return frame

    @classmethod
    def _get_map_to_room_map(
        cls,
        *,
        room_json_path: Optional[Union[str, Path]] = None,
    ) -> Dict[str, str]:
        if cls._map_to_room_map is not None and room_json_path is None:
            return cls._map_to_room_map

        explicit = build_map_to_room_map(room_json_path)
        if explicit:
            cls._map_to_room_map = explicit
            return cls._map_to_room_map

        if cls._map_to_room_map is None:
            cls._map_to_room_map = {}
        return cls._map_to_room_map

    def similarity_check(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
        *,
        mode: str = "pose",
        trans_tol_m: float = 3.0,
        rot_tol_deg: float = 60.0,
    ) -> bool:
        room_a = a.get("room", None)
        room_b = b.get("room", None)
        if room_a is None or room_b is None or room_a != room_b:
            return False

        if mode == "room" and room_a == room_b:
            return True

        pose_a = a["pose"]
        pose_b = b["pose"]
        if not isinstance(pose_a, Tensor) or not isinstance(pose_b, Tensor):
            pose_a = torch.as_tensor(pose_a)
            pose_b = torch.as_tensor(pose_b)

        pose_a = pose_a.to(dtype=torch.float64)
        pose_b = pose_b.to(dtype=torch.float64)
        if pose_a.numel() != 7 or pose_b.numel() != 7:
            raise ValueError(
                f"Expected pose vectors of length 7; got {pose_a.numel()} and {pose_b.numel()}"
            )

        trans_diff = float((pose_a[:3] - pose_b[:3]).norm())
        if trans_diff > float(trans_tol_m):
            return False

        rot_diff_deg = quaternion_angle(
            pose_a[3:].detach().cpu().numpy(),
            pose_b[3:].detach().cpu().numpy(),
            degrees=True,
            normalize=True,
        )
        return rot_diff_deg <= float(rot_tol_deg)

    def collate_fn(self, batch: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "idxs": torch.stack([b["idx"] for b in batch], dim=0),
            "poses": torch.stack([b["pose"] for b in batch], dim=0),
            "scenes_hashes": torch.stack([b["scene_hash"] for b in batch], dim=0),
        }
        if "image_main" in batch[0]:
            out["images_main"] = torch.stack([b["image_main"] for b in batch], dim=0)
        if "graph_main" in batch[0]:
            out["graphs_main"] = _collate_graph_objects(
                [b["graph_main"] for b in batch],
                feat_dim=self.graph_feat_dim,
                feat_edge_attr_dim=self.graph_edge_attr_dim,
            )
        return out
