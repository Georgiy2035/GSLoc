"""Replica dataset loader and metadata builder."""

from __future__ import annotations

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
from gsloc.utils.graphs import _collate_graph_objects, _ensure_nonempty, _sanitize_graph_obj, rotate_graph_features
from mmpr.modules.vis_utils import quaternion_angle
from opr.datasets.augmentations import DefaultImageTransform


def _read_pose_matrix(pose_path: Path) -> np.ndarray:
    """Read a 4x4 pose matrix from a Replica pose text file."""
    mat = np.loadtxt(str(pose_path), dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected pose matrix (4, 4), got {mat.shape} for {pose_path}")
    return mat


def _matrix_to_pose7(T_wc: np.ndarray) -> List[float]:
    """Convert world-from-camera transform to [tx, ty, tz, qx, qy, qz, qw]."""
    t = T_wc[:3, 3].astype(np.float64)
    q_xyzw = R.from_matrix(T_wc[:3, :3]).as_quat().astype(np.float64)
    return [float(t[0]), float(t[1]), float(t[2]), float(q_xyzw[0]), float(q_xyzw[1]), float(q_xyzw[2]), float(q_xyzw[3])]


def _load_scene_ids(scene_list_path: Union[str, Path]) -> Set[str]:
    """Load scene IDs from a newline-separated text file."""
    scene_list_path = Path(scene_list_path)
    if not scene_list_path.exists():
        raise FileNotFoundError(f"Missing scene list file: {scene_list_path}")
    return {line.strip() for line in scene_list_path.read_text(encoding="utf-8").splitlines() if line.strip()}


def resolve_replica_scene_filter(
    *,
    scene_list_path: Optional[Union[str, Path]] = None,
    scene_filter_mode: str = "all",
) -> Optional[Set[str]]:
    """Resolve selected Replica scene IDs.

    Modes:
    - `all`: keep all scenes
    - `listed`: keep only scenes from `scene_list_path`
    """
    if scene_filter_mode == "all":
        return None
    if scene_filter_mode == "listed":
        if scene_list_path is None:
            raise ValueError("`scene_list_path` must be provided when `scene_filter_mode='listed'`")
        return _load_scene_ids(scene_list_path)
    raise ValueError("`scene_filter_mode` must be one of: 'all', 'listed'")


def iter_replica_frames(
    dataset_root: Union[str, Path],
    modality: list[str] = ["image"],
    graph_dir: str = "SceneGraphs_replica_pt_compact",
) -> Iterable[Dict[str, Any]]:
    """Yield frame records for Replica layout: data/<scene>/sequence/frame-*.color.jpg."""
    root = Path(dataset_root)
    scenes_root = root / "data"
    if not scenes_root.exists():
        raise FileNotFoundError(f"Expected scenes folder at {scenes_root}")

    for scene_dir in sorted(p for p in scenes_root.iterdir() if p.is_dir()):
        scene_id = scene_dir.name
        seq_dir = scene_dir / "sequence"
        if not seq_dir.exists():
            continue
        for pose_path in sorted(seq_dir.glob("*.pose.txt")):
            frame = {
                "scene_id": scene_id,
                "pose_path": pose_path,
            }
            if "image" in modality:
                image_path = pose_path.with_suffix("").with_suffix(".color.jpg")
                if not image_path.exists():
                    continue
                frame["image_path"] = image_path
            if "graph" in modality:
                graph_path = root / graph_dir / scene_id / f"{pose_path.with_suffix('').with_suffix('').name}.pt"
                if not graph_path.exists():
                    continue
                frame["graph_path"] = graph_path
            yield frame


def build_replica_df(
    dataset_root: Union[str, Path],
    *,
    limit: Optional[int] = None,
    log_every: int = 50_000,
    modality: list[str] = ["image"],
    scene_ids: Optional[Set[str]] = None,
    graph_dir: str = "SceneGraphs_replica_pt_compact",
) -> pd.DataFrame:
    """Build metadata dataframe for Replica, optionally restricted to scene IDs."""
    root = Path(dataset_root)
    rows: List[Dict[str, Any]] = []
    n = 0

    if scene_ids is None:
        logger.info("Scanning Replica dataset...")
    else:
        logger.info(f"Scanning Replica dataset for {len(scene_ids)} selected scenes...")

    for frame in iter_replica_frames(root, modality=modality, graph_dir=graph_dir):
        if scene_ids is not None and frame["scene_id"] not in scene_ids:
            continue
        try:
            T_wc = _read_pose_matrix(frame["pose_path"])
            pose7 = _matrix_to_pose7(T_wc)
        except Exception:
            logger.exception(f"Failed reading pose for {frame['pose_path']}")
            continue

        row: Dict[str, Any] = {
            "idx": n,
            "scene": frame["scene_id"],
            "pose": pose7,
        }
        if "image" in modality:
            row["image_path"] = str(frame["image_path"])
        if "graph" in modality:
            row["graph_path"] = str(frame["graph_path"])
        rows.append(row)

        n += 1
        if log_every and (n % log_every == 0):
            logger.info(f"Scanned {n:,} frames...")
        if limit is not None and n >= limit:
            break

    if not rows:
        raise RuntimeError(f"No frames with poses found under {root}/data/*/sequence")

    df = pd.DataFrame(rows)
    logger.info(f"Scanned {n} rows")
    return df


class Replica(PRDataset):
    """Replica dataset with ThreeRScan-compatible interface."""

    def __init__(
        self,
        dataset_root: Union[str, Path] = "/mnt/external_usb_hdd/6YL/Datasets/Replica",
        *,
        meta_path: Optional[Union[str, Path]] = None,
        meta_file: str = "meta.parquet",
        modality: list[str] = ["image"],
        rebuild_meta: bool = False,
        save_meta: bool = False,
        limit: Optional[int] = None,
        scene_list_path: Optional[Union[str, Path]] = None,
        scene_filter_mode: str = "all",
        image_transform: Any = DefaultImageTransform(resize=(320, 192), train=False),
        graph_feat_dim: int = 4,
        graph_edge_attr_dim: int = 7,
        graph_rotate: bool = True,
        graph_dir: str = "SceneGraphs_replica_pt_compact",
    ) -> None:
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.modality = modality
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Given dataset_root={self.dataset_root} doesn't exist")

        self.meta_path = self.dataset_root / meta_file if meta_path is None else Path(meta_path) / meta_file
        if scene_list_path is None and scene_filter_mode != "all":
            scene_list_path = self.dataset_root / "files" / "test_scenes.txt"
        self.scene_list_path = Path(scene_list_path) if scene_list_path is not None else None
        selected_scene_ids = resolve_replica_scene_filter(
            scene_list_path=self.scene_list_path,
            scene_filter_mode=scene_filter_mode,
        )

        can_use_cached_meta = self.meta_path.exists() and not rebuild_meta
        if can_use_cached_meta:
            self.df = pd.read_parquet(self.meta_path)
        else:
            logger.info(
                "Rebuilding metadata for Replica dataset"
                if self.meta_path.exists()
                else "Metadata not found, rebuilding metadata for Replica dataset"
            )
            self.df = build_replica_df(
                self.dataset_root,
                limit=limit,
                modality=modality,
                scene_ids=selected_scene_ids,
                graph_dir=graph_dir,
            )

        if selected_scene_ids is not None and can_use_cached_meta:
            self.df = self.df[self.df["scene"].astype(str).isin(selected_scene_ids)].reset_index(drop=True)
            self.df["idx"] = np.arange(len(self.df), dtype=np.int64)

        if limit is not None and can_use_cached_meta:
            self.df = self.df.iloc[:limit].reset_index(drop=True)
            self.df["idx"] = np.arange(len(self.df), dtype=np.int64)

        required_cols = {"idx", "pose", "scene"}
        if "image" in modality:
            required_cols.add("image_path")
        if "graph" in modality:
            required_cols.add("graph_path")
        missing_cols = required_cols - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"{self.meta_path} is missing columns: {sorted(missing_cols)}")

        self.image_transform = image_transform
        self.graph_feat_dim = graph_feat_dim
        self.graph_edge_attr_dim = graph_edge_attr_dim
        self.graph_rotate = graph_rotate

        if save_meta:
            self.save_meta_parquet(self.meta_path.parent, meta_file)

    def _load_image(self, image_path: Union[str, Path]) -> Tensor:
        img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(f"Failed to read image: {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.image_transform is not None:
            img = self.image_transform(img)
        if not isinstance(img, torch.Tensor):
            img = torch.from_numpy(np.ascontiguousarray(img))
            if img.ndim == 3:
                img = img.permute(2, 0, 1)
            img = img.float() / 255.0
        return img

    def _load_graph(self, graph_path: Union[str, Path, None]) -> Any:
        if graph_path is None:
            return None
        graph = torch.load(graph_path, map_location="cpu", weights_only=False)
        graph = _sanitize_graph_obj(graph, feat_dim=self.graph_feat_dim, feat_edge_attr_dim=self.graph_edge_attr_dim)

        if isinstance(graph, list):
            out = []
            for g in graph:
                g = _ensure_nonempty(g, self.graph_feat_dim, self.graph_edge_attr_dim)
                if self.graph_rotate:
                    g = rotate_graph_features(g)
                out.append(g)
            return out

        graph = _ensure_nonempty(graph, self.graph_feat_dim, self.graph_edge_attr_dim)
        if self.graph_rotate:
            graph = rotate_graph_features(graph)
        return graph

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[int(idx)]
        scene = str(row["scene"])
        pose = torch.tensor(np.asarray(row["pose"], dtype=np.float32), dtype=torch.float32)

        frame: Dict[str, Any] = {
            "idx": torch.tensor(int(row["idx"]), dtype=torch.int64),
            "scene": scene,
            "scene_hash": torch.tensor(hash(scene)),
            "pose": pose,
        }
        if "image" in self.modality:
            frame["image_main"] = self._load_image(row["image_path"])
        if "graph" in self.modality:
            frame["graph_main"] = self._load_graph(row["graph_path"])
        return frame

    def similarity_check(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
        *,
        trans_tol_m: float = 3.0,
        rot_tol_deg: float = 60.0,
    ) -> bool:
        """Check same-scene and close pose criteria for positive pairs."""
        scene_a = a.get("scene", None)
        scene_b = b.get("scene", None)
        if scene_a is None or scene_b is None or scene_a != scene_b:
            return False

        pose_a = a["pose"]
        pose_b = b["pose"]
        if not isinstance(pose_a, Tensor) or not isinstance(pose_b, Tensor):
            pose_a = torch.as_tensor(pose_a)
            pose_b = torch.as_tensor(pose_b)

        pose_a = pose_a.to(dtype=torch.float64)
        pose_b = pose_b.to(dtype=torch.float64)
        if pose_a.numel() != 7 or pose_b.numel() != 7:
            raise ValueError(f"Expected pose vectors of length 7; got {pose_a.numel()} and {pose_b.numel()}")

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
        """Collate function compatible with image-only and image+graph modalities."""
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



