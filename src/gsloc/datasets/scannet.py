"""ScanNet dataset loader and metadata builder.

This module provides:
- ``build_scannet_df``: scans the dataset, loads per-frame poses, and builds a metadata dataframe.
- ``ScanNet``: a PyTorch ``Dataset`` with a ThreeRScan-compatible PR interface.

Dataset layout (under ``/mnt/external_usb_hdd/6YL/Datasets/ScanNet``):

``<scans_dir>/<scene_id>/sens/color/<frame_id>.jpg``
``<scans_dir>/<scene_id>/sens/pose/<frame_id>.txt``  (4x4 matrix, one row per line)
``makarov_images/<scene_id>/<frame_id>.jpg``  (optional subsampled RGB)
``SceneGraphs_Makarov/<scene_id>/<frame_id>.json``  (or ``.pt``)

Scenes that belong to the same physical room share the room prefix in their ids,
e.g. ``scene0000_00`` and ``scene0000_01`` both map to room ``scene0000``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union

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
    convert_scenegraph_json_to_compact_pt,
    rotate_graph_features,
)
from mmpr.modules.vis_utils import quaternion_angle
from opr.datasets.augmentations import DefaultImageTransform

DEFAULT_DATASET_ROOT = Path("/mnt/external_usb_hdd/6YL/Datasets/ScanNet")
DEFAULT_SCANS_DIR = "scans"
DEFAULT_IMAGE_DIR = "makarov_images"
DEFAULT_GRAPH_REL_PATH = Path("SceneGraphs_Makarov")


def _read_pose_matrix(pose_path: Path) -> np.ndarray:
    """Read a 4x4 pose matrix from a ScanNet ``*.txt`` pose file."""
    mat = np.loadtxt(str(pose_path), dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected pose matrix (4,4), got {mat.shape} for {pose_path}")
    return mat


def _matrix_to_pose7(T_wc: np.ndarray) -> List[float]:
    """Convert 4x4 world-from-camera matrix to [tx, ty, tz, qx, qy, qz, qw]."""
    t = T_wc[:3, 3].astype(np.float64)
    q_xyzw = R.from_matrix(T_wc[:3, :3]).as_quat().astype(np.float64)
    return [
        float(t[0]),
        float(t[1]),
        float(t[2]),
        float(q_xyzw[0]),
        float(q_xyzw[1]),
        float(q_xyzw[2]),
        float(q_xyzw[3]),
    ]


def scene_id_to_room_id(scene_id: str) -> str:
    """Map ScanNet scene id to room id (``scene0000_00`` -> ``scene0000``)."""
    parts = str(scene_id).rsplit("_", 1)
    if len(parts) == 2 and len(parts[1]) == 2 and parts[1].isdigit():
        return parts[0]
    return str(scene_id)


def _frame_id_sort_key(path: Path) -> int:
    return int(path.stem)


def _load_scene_ids(scene_list_path: Union[str, Path]) -> Set[str]:
    scene_list_path = Path(scene_list_path)
    if not scene_list_path.exists():
        raise FileNotFoundError(f"Missing scene list file: {scene_list_path}")
    return {
        line.strip()
        for line in scene_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build_scene_to_room_map(
    dataset_root: Union[str, Path],
    *,
    scans_dir: Union[str, Path] = DEFAULT_SCANS_DIR,
    scene_ids: Optional[Set[str]] = None,
) -> Dict[str, str]:
    """Build ``scene_id -> room_id`` mapping from ScanNet scene directory names."""
    scenes_root = Path(dataset_root) / scans_dir
    if not scenes_root.exists():
        raise FileNotFoundError(f"Expected scenes folder at {scenes_root}")

    scene_to_room: Dict[str, str] = {}
    for scene_dir in sorted(p for p in scenes_root.iterdir() if p.is_dir()):
        scene_id = scene_dir.name
        if scene_ids is not None and scene_id not in scene_ids:
            continue
        scene_to_room[scene_id] = scene_id_to_room_id(scene_id)

    if not scene_to_room:
        raise RuntimeError(f"Failed to build scene->room mapping from {scenes_root}")
    return scene_to_room


def resolve_scannet_scene_filter(
    *,
    scene_list_path: Optional[Union[str, Path]] = None,
    scene_filter_mode: str = "all",
    dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
    scans_dir: Union[str, Path] = DEFAULT_SCANS_DIR,
) -> Optional[Set[str]]:
    """
    Resolve a set of scene ids to keep.

    Modes:
    - ``all``: keep all scenes.
    - ``listed``: keep only scenes listed in ``scene_list_path``.
    - ``same_room_excluding_listed``: keep all scenes from rooms that contain a listed scene,
      but exclude the listed scenes themselves.
    """
    if scene_filter_mode == "all":
        return None

    if scene_list_path is None:
        raise ValueError("`scene_list_path` must be provided when `scene_filter_mode` is not 'all'")

    listed_scene_ids = _load_scene_ids(scene_list_path)
    if scene_filter_mode == "listed":
        return listed_scene_ids

    if scene_filter_mode == "same_room_excluding_listed":
        scene_to_room = build_scene_to_room_map(dataset_root, scans_dir=scans_dir)
        listed_rooms = {
            scene_to_room[scene_id] for scene_id in listed_scene_ids if scene_id in scene_to_room
        }
        if not listed_rooms:
            raise RuntimeError(
                f"None of the listed scenes from {scene_list_path} were found under "
                f"{Path(dataset_root) / scans_dir}"
            )
        return {
            scene_id
            for scene_id, room_id in scene_to_room.items()
            if room_id in listed_rooms and scene_id not in listed_scene_ids
        }

    raise ValueError(
        "`scene_filter_mode` must be one of: 'all', 'listed', 'same_room_excluding_listed'"
    )


def _resolve_image_path(
    dataset_root: Path,
    *,
    scene_id: str,
    frame_id: str,
    scans_dir: Union[str, Path],
    image_dir: Optional[Union[str, Path]],
) -> Optional[Path]:
    if image_dir is not None:
        image_path = dataset_root / image_dir / scene_id / f"{frame_id}.jpg"
        if image_path.is_file():
            return image_path

    sens_image = dataset_root / scans_dir / scene_id / "sens" / "color" / f"{frame_id}.jpg"
    if sens_image.is_file():
        return sens_image
    return None


def _resolve_graph_path(
    dataset_root: Path,
    *,
    scene_id: str,
    frame_id: str,
    graph_path: Optional[Union[str, Path]],
) -> Optional[Path]:
    if graph_path is None:
        return None
    graph_root = dataset_root / graph_path / scene_id
    for ext in (".pt", ".json"):
        candidate = graph_root / f"{frame_id}{ext}"
        if candidate.is_file():
            return candidate
    return None


def _iter_scene_frame_ids(
    dataset_root: Path,
    scene_id: str,
    *,
    scans_dir: Union[str, Path],
    modality: list[str],
    graph_path: Optional[Union[str, Path]],
) -> Iterable[str]:
    if "graph" in modality and graph_path is not None:
        graph_scene_dir = dataset_root / graph_path / scene_id
        if graph_scene_dir.is_dir():
            graph_files = sorted(
                list(graph_scene_dir.glob("*.json")) + list(graph_scene_dir.glob("*.pt")),
                key=_frame_id_sort_key,
            )
            for graph_file in graph_files:
                yield graph_file.stem
            return

    pose_dir = dataset_root / scans_dir / scene_id / "sens" / "pose"
    if not pose_dir.is_dir():
        return
    for pose_path in sorted(pose_dir.glob("*.txt"), key=_frame_id_sort_key):
        yield pose_path.stem


def iter_scannet_frames(
    dataset_root: Union[str, Path],
    *,
    scans_dir: Union[str, Path] = DEFAULT_SCANS_DIR,
    image_dir: Optional[Union[str, Path]] = DEFAULT_IMAGE_DIR,
    modality: list[str] = ["image", "graph"],
    graph_path: Optional[Union[str, Path]] = DEFAULT_GRAPH_REL_PATH,
) -> Iterable[Dict[str, Any]]:
    """Yield per-frame records for ScanNet scenes."""
    root = Path(dataset_root)
    scenes_root = root / scans_dir
    if not scenes_root.exists():
        raise FileNotFoundError(f"Expected scenes folder at {scenes_root}")

    for scene_dir in sorted(p for p in scenes_root.iterdir() if p.is_dir()):
        scene_id = scene_dir.name
        for frame_id in _iter_scene_frame_ids(
            root,
            scene_id,
            scans_dir=scans_dir,
            modality=modality,
            graph_path=graph_path,
        ):
            frame: Dict[str, Any] = {"scene_id": scene_id, "frame_id": frame_id}

            pose_path = root / scans_dir / scene_id / "sens" / "pose" / f"{frame_id}.txt"
            if not pose_path.is_file():
                continue
            frame["pose_path"] = pose_path

            if "image" in modality:
                image_path = _resolve_image_path(
                    root,
                    scene_id=scene_id,
                    frame_id=frame_id,
                    scans_dir=scans_dir,
                    image_dir=image_dir,
                )
                if image_path is None:
                    continue
                frame["image_path"] = image_path

            if "graph" in modality:
                graph_file = _resolve_graph_path(
                    root,
                    scene_id=scene_id,
                    frame_id=frame_id,
                    graph_path=graph_path,
                )
                if graph_file is None:
                    continue
                frame["graph_path"] = graph_file

            yield frame


def build_scannet_df(
    dataset_root: Union[str, Path],
    *,
    scans_dir: Union[str, Path] = DEFAULT_SCANS_DIR,
    image_dir: Optional[Union[str, Path]] = DEFAULT_IMAGE_DIR,
    limit: Optional[int] = None,
    log_every: int = 50_000,
    modality: list[str] = ["image", "graph"],
    scene_ids: Optional[Set[str]] = None,
    scene_to_room_map: Optional[Dict[str, str]] = None,
    graph_path: Optional[Union[str, Path]] = DEFAULT_GRAPH_REL_PATH,
    similarity_filter_mode: str = "none",
    similarity_trans_tol_m: float = 3.0,
    similarity_rot_tol_deg: float = 60.0,
) -> pd.DataFrame:
    """Build a metadata dataframe for ScanNet, optionally restricted to a scene subset."""
    root = Path(dataset_root)

    rows: List[Dict[str, Any]] = []
    n = 0
    last_kept_by_scene: Dict[str, Dict[str, Any]] = {}
    skipped_similar = 0

    if similarity_filter_mode not in {"none", "pose", "room"}:
        raise ValueError("`similarity_filter_mode` must be one of: 'none', 'pose', 'room'")

    if scene_ids is None:
        logger.info("Scanning ScanNet dataset...")
    else:
        logger.info("Scanning ScanNet dataset for {} selected scenes...", len(scene_ids))

    for frame in iter_scannet_frames(
        root,
        scans_dir=scans_dir,
        image_dir=image_dir,
        modality=modality,
        graph_path=graph_path,
    ):
        if scene_ids is not None and frame["scene_id"] not in scene_ids:
            continue
        try:
            T_wc = _read_pose_matrix(frame["pose_path"])
            pose7 = _matrix_to_pose7(T_wc)
        except Exception:
            logger.exception("Failed reading pose for {}", frame["pose_path"])
            continue

        row: Dict[str, Any] = {
            "idx": n,
            "scene": frame["scene_id"],
            "room": (scene_to_room_map or {}).get(frame["scene_id"], scene_id_to_room_id(frame["scene_id"])),
            "pose": pose7,
            "frame_id": frame["frame_id"],
        }
        if "image" in modality:
            row["image_path"] = str(frame["image_path"])
        if "graph" in modality:
            row["graph_path"] = str(frame["graph_path"])

        if similarity_filter_mode != "none":
            scene_id = str(row["scene"])
            prev = last_kept_by_scene.get(scene_id)
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

                        t_a = pose_a[:3]
                        t_b = pose_b[:3]
                        trans_diff = float((t_a - t_b).norm())

                        if trans_diff <= float(similarity_trans_tol_m):
                            q_a = pose_a[3:]
                            q_b = pose_b[3:]
                            rot_diff_deg = quaternion_angle(
                                q_a.detach().cpu().numpy(),
                                q_b.detach().cpu().numpy(),
                                degrees=True,
                                normalize=True,
                            )
                            is_similar = rot_diff_deg <= float(similarity_rot_tol_deg)

                if is_similar:
                    skipped_similar += 1
                    continue

            last_kept_by_scene[scene_id] = {
                "room": row["room"],
                "pose": row["pose"],
            }

        rows.append(row)
        n += 1
        if log_every and (n % log_every == 0):
            logger.info("Scanned {:,} frames...", n)
        if limit is not None and n >= limit:
            break

    if not rows:
        if scene_ids is None:
            raise RuntimeError(f"No frames with poses found under {root}/{scans_dir}/*/sens/pose")
        raise RuntimeError("No frames with poses found for the requested scene selection")

    df = pd.DataFrame(rows)
    logger.info("Scanned {} rows", n)
    if similarity_filter_mode != "none":
        logger.info(
            "Similarity filtering skipped {} sequential frames (mode={})",
            skipped_similar,
            similarity_filter_mode,
        )
    return df


class ScanNet(PRDataset):
    """ScanNet dataset with ThreeRScan-compatible PR interface."""

    _scene_to_room_map: Optional[Dict[str, str]] = None

    def __init__(
        self,
        dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
        *,
        meta_path: Optional[Union[str, Path]] = None,
        meta_file: str = "meta.parquet",
        scans_dir: Union[str, Path] = DEFAULT_SCANS_DIR,
        image_dir: Optional[Union[str, Path]] = DEFAULT_IMAGE_DIR,
        modality: list[str] = ["image", "graph"],
        rebuild_meta: bool = False,
        save_meta: bool = False,
        limit: Optional[int] = None,
        scene_list_path: Optional[Union[str, Path]] = None,
        scene_filter_mode: str = "all",
        image_transform: Any = DefaultImageTransform(resize=(320, 192), train=False),
        graph_feat_dim: int = 4,
        graph_edge_attr_dim: int = 10,
        graph_rotate: bool = True,
        edge_normalizer_path: Optional[Union[str, Path]] = None,
        graph_path: Optional[Union[str, Path]] = DEFAULT_GRAPH_REL_PATH,
        similarity_filter_mode: str = "none",
        similarity_trans_tol_m: float = 1.0,
        similarity_rot_tol_deg: float = 15.0,
        room_json_path: Optional[Union[str, Path]] = None,  # kept for TestConfig compatibility; unused
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.modality = modality
        self.scans_dir = Path(scans_dir)
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.graph_path = Path(graph_path) if graph_path is not None else None

        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Given dataset_root={self.dataset_root} doesn't exist")

        self.meta_path = self.dataset_root / meta_file if meta_path is None else Path(meta_path) / meta_file
        self.scene_list_path = Path(scene_list_path) if scene_list_path is not None else None
        self.scene_filter_mode = scene_filter_mode
        self.similarity_filter_mode = similarity_filter_mode
        self.similarity_trans_tol_m = float(similarity_trans_tol_m)
        self.similarity_rot_tol_deg = float(similarity_rot_tol_deg)

        if room_json_path is not None:
            logger.warning(
                "ScanNet uses scene-name room grouping; room_json_path={} is ignored",
                room_json_path,
            )

        selected_scene_ids = resolve_scannet_scene_filter(
            scene_list_path=self.scene_list_path,
            scene_filter_mode=self.scene_filter_mode,
            dataset_root=self.dataset_root,
            scans_dir=self.scans_dir,
        )

        can_use_cached_meta = self.meta_path.exists() and not rebuild_meta
        if can_use_cached_meta:
            self.df = pd.read_parquet(self.meta_path)
        else:
            logger.info(
                "Rebuilding metadata for ScanNet dataset"
                if self.meta_path.exists()
                else "Metadata not found, rebuilding metadata for ScanNet dataset"
            )
            self.df = build_scannet_df(
                self.dataset_root,
                scans_dir=self.scans_dir,
                image_dir=self.image_dir,
                limit=limit,
                modality=modality,
                scene_ids=selected_scene_ids,
                scene_to_room_map=self._get_scene_to_room_map(scans_dir=self.scans_dir),
                graph_path=self.graph_path,
                similarity_filter_mode=self.similarity_filter_mode,
                similarity_trans_tol_m=self.similarity_trans_tol_m,
                similarity_rot_tol_deg=self.similarity_rot_tol_deg,
            )

        if selected_scene_ids is not None and can_use_cached_meta:
            self.df = self.df[self.df["scene"].astype(str).isin(selected_scene_ids)].reset_index(drop=True)
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
            raise ValueError(f"{self.meta_path} is missing columns: {sorted(missing_cols)}")

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
            logger.warning("Edge normalizer path is None; graph edge_attr will not be normalized")

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
                logger.warning("Missing or unreadable {} {}: {} — {}", kind, path, type(exc).__name__, exc)
            else:
                logger.warning("Missing or unreadable {} {}", kind, path)
        elif self._missing_asset_warn_count == 26:
            logger.warning("Further missing image/graph file warnings suppressed (first 25 were logged)")

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
            if p.suffix == ".json":
                graph = convert_scenegraph_json_to_compact_pt(p, graph_rotated=True)
            else:
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
                    "Skipping edge normalization: normalizer stats dim {} != edge_attr dim {} "
                    "(log once per dataset instance; edge_attr left unnormalized)",
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

        frame = {
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
    def _get_scene_to_room_map(
        cls,
        *,
        scans_dir: Union[str, Path] = DEFAULT_SCANS_DIR,
        dataset_root: Union[str, Path] = DEFAULT_DATASET_ROOT,
    ) -> Dict[str, str]:
        if cls._scene_to_room_map is not None:
            return cls._scene_to_room_map

        cls._scene_to_room_map = build_scene_to_room_map(dataset_root, scans_dir=scans_dir)
        return cls._scene_to_room_map

    def similarity_check(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
        *,
        mode: str = "pose",
        trans_tol_m: float = 3.0,
        rot_tol_deg: float = 60.0,
    ) -> bool:
        """Return True if both samples belong to the same room and pose is within tolerance."""
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

        t_a = pose_a[:3]
        t_b = pose_b[:3]
        trans_diff = float((t_a - t_b).norm())
        if trans_diff > float(trans_tol_m):
            return False

        q_a = pose_a[3:]
        q_b = pose_b[3:]
        rot_diff_deg = quaternion_angle(
            q_a.detach().cpu().numpy(),
            q_b.detach().cpu().numpy(),
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
        if "image_main" in batch[0].keys():
            out["images_main"] = torch.stack([b["image_main"] for b in batch], dim=0)
        if "graph_main" in batch[0].keys():
            out["graphs_main"] = _collate_graph_objects(
                [b["graph_main"] for b in batch],
                feat_dim=self.graph_feat_dim,
                feat_edge_attr_dim=self.graph_edge_attr_dim,
            )
        return out
