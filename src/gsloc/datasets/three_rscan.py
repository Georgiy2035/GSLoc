"""3RScan dataset loader and metadata builder.

This module provides:
- `build_3rscan_meta_parquet`: scans the dataset, loads per-frame poses, and writes a single `meta.parquet`.
- `ThreeRScan`: a PyTorch `Dataset` that loads RGB images from all scenes using the `meta.parquet`.

Dataset layout assumed (matches your dataset under `/mnt/external_usb_hdd/6YL/Datasets/3RScan`):

`scenes/<scene_id>/sequence/frame-XXXXXX.color.jpg`
`scenes/<scene_id>/sequence/frame-XXXXXX.pose.txt`  (4x4 matrix, one row per line)
"""

from __future__ import annotations

import json
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

from opr.datasets.augmentations import DefaultImageTransform
from mmpr.modules.vis_utils import quaternion_angle
from gsloc.utils.graphs import _sanitize_graph_obj, _ensure_nonempty, rotate_graph_features, _collate_graph_objects


def _read_pose_matrix(pose_path: Path) -> np.ndarray:
    """Read a 4x4 pose matrix from a 3RScan `*.pose.txt` file."""
    mat = np.loadtxt(str(pose_path), dtype=np.float64)
    if mat.shape != (4, 4):
        raise ValueError(f"Expected pose matrix (4,4), got {mat.shape} for {pose_path}")
    return mat


def _matrix_to_pose7(T_wc: np.ndarray) -> List[float]:
    """Convert 4x4 world-from-camera matrix to [tx, ty, tz, qx, qy, qz, qw]."""
    t = T_wc[:3, 3].astype(np.float64)
    q_xyzw = R.from_matrix(T_wc[:3, :3]).as_quat().astype(np.float64)  # x,y,z,w
    return [
        float(t[0]),
        float(t[1]),
        float(t[2]),
        float(q_xyzw[0]),
        float(q_xyzw[1]),
        float(q_xyzw[2]),
        float(q_xyzw[3]),
    ]


def iter_3rscan_frames(
    dataset_root: Union[str, Path], 
    modality: list[str] = ["image", "graph"]
    ) -> Iterable[Tuple[str, Path, Path]]:

    """Yield (scene_id, image_path, pose_path) for all frames that have a pose."""
    root = Path(dataset_root)
    scenes_root = root / "scenes"
    if not scenes_root.exists():
        raise FileNotFoundError(f"Expected scenes folder at {scenes_root}")

    for scene_dir in sorted(p for p in scenes_root.iterdir() if p.is_dir()):
        scene_id = scene_dir.name
        seq_dir = scene_dir / "sequence"
        if not seq_dir.exists():
            continue
        frame = dict()
        for pose_path in sorted(seq_dir.glob("*.pose.txt")):
            frame["pose_path"] = pose_path
            frame["scene_id"] = scene_id
            if "image" in modality:
                frame["image_path"] = pose_path.with_suffix("").with_suffix(".color.jpg")  # frame-XXXXXX.pose.txt
            if "graph" in modality:
                frame["graph_path"] = root / "Splited_graphs" / scene_id / (pose_path.with_suffix("").with_suffix("").name + ".pt")
            if all(frame.values()):
                yield frame


def _load_scene_ids(scene_list_path: Union[str, Path]) -> Set[str]:
    """Load a newline-separated scene list file."""
    scene_list_path = Path(scene_list_path)
    if not scene_list_path.exists():
        raise FileNotFoundError(f"Missing scene list file: {scene_list_path}")
    return {
        line.strip()
        for line in scene_list_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def build_scene_to_room_map(
    room_json_path: Union[str, Path] = "/mnt/external_usb_hdd/6YL/Datasets/3RScan/files/3RScan.json",
) -> Dict[str, str]:
    """
    Build mapping `scene_id -> room_reference`.

    In `3RScan.json`, scenes that belong to the same physical room are grouped under the same
    top-level object, with the top-level `"reference"` and all entries from `"scans"[]."reference"`.
    """
    room_json_path = Path(room_json_path)
    if not room_json_path.exists():
        raise FileNotFoundError(f"Missing 3RScan room file: {room_json_path}")

    with room_json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError(f"Expected a JSON list at top-level in {room_json_path}")

    scene_to_room: Dict[str, str] = {}
    for entry in data:
        if not isinstance(entry, dict):
            continue
        room_ref = entry.get("reference")
        scans = entry.get("scans")
        if room_ref is None or not isinstance(scans, list):
            continue

        for scan_entry in [{"reference": room_ref}] + scans:
            if not isinstance(scan_entry, dict):
                continue
            scan_ref = scan_entry.get("reference")
            if scan_ref is None:
                continue
            scene_to_room[str(scan_ref)] = str(room_ref)

    if not scene_to_room:
        raise RuntimeError(f"Failed to build scene->room mapping from {room_json_path}")

    return scene_to_room


def resolve_3rscan_scene_filter(
    *,
    scene_list_path: Optional[Union[str, Path]] = None,
    scene_filter_mode: str = "all",
    room_json_path: Union[str, Path] = "/mnt/external_usb_hdd/6YL/Datasets/3RScan/files/3RScan.json",
) -> Optional[Set[str]]:
    """
    Resolve a set of scene ids to keep.

    Modes:
    - `all`: keep all scenes.
    - `listed`: keep only scenes listed in `scene_list_path`.
    - `same_room_excluding_listed`: keep all scenes from rooms that contain a listed scene,
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
        scene_to_room = build_scene_to_room_map(room_json_path=room_json_path)
        listed_rooms = {scene_to_room[scene_id] for scene_id in listed_scene_ids if scene_id in scene_to_room}
        if not listed_rooms:
            raise RuntimeError(
                f"None of the listed scenes from {scene_list_path} were found in {room_json_path}"
            )
        return {
            scene_id
            for scene_id, room_id in scene_to_room.items()
            if room_id in listed_rooms and scene_id not in listed_scene_ids
        }

    raise ValueError(
        "`scene_filter_mode` must be one of: 'all', 'listed', 'same_room_excluding_listed'"
    )


def build_3rscan_df(
    dataset_root: Union[str, Path],
    *,
    limit: Optional[int] = None,
    log_every: int = 50_000,
    modality: list[str] = ["image", "graph"],
    scene_ids: Optional[Set[str]] = None,
    scene_to_room_map: Optional[Dict[str, str]] = None,
) -> pd.DataFrame:
    """Build a metadata dataframe for 3RScan, optionally restricted to a scene subset."""
    root = Path(dataset_root)

    rows: List[Dict[str, Any]] = []
    n = 0

    if scene_ids is None:
        logger.info("Scanning 3rscan dataset...")
    else:
        logger.info(f"Scanning 3rscan dataset for {len(scene_ids)} selected scenes...")

    for frame in iter_3rscan_frames(root, modality=modality):
        if scene_ids is not None and frame["scene_id"] not in scene_ids:
            continue
        try:
            T_wc = _read_pose_matrix(frame["pose_path"])
            pose7 = _matrix_to_pose7(T_wc)
        except Exception:
            logger.exception(f"Failed reading pose for {frame['pose_path']}")
            continue


        row = dict()
        row["idx"] = n
        row["scene"] = frame["scene_id"]
        row["room"] = scene_to_room_map.get(frame["scene_id"], None)
        row["pose"] = pose7
        if "image" in modality:
            row["image_path"] = str(frame.get("image_path", None))
        if "graph" in modality:
            row["graph_path"] = str(frame.get("graph_path", None))
        rows.append(row)
        n += 1
        if log_every and (n % log_every == 0):
            logger.info(f"Scanned {n:,} frames...")
        if limit is not None and n >= limit:
            break

    if not rows:
        if scene_ids is None:
            raise RuntimeError(f"No frames with poses found under {root}/scenes/*/sequence")
        raise RuntimeError("No frames with poses found for the requested scene selection")

    df = pd.DataFrame(rows)
    logger.info(f"Scanned {n} rows")

    return df
        

class ThreeRScan(PRDataset):
    """3RScan dataset that loads RGB frames from all scenes."""

    _scene_to_room_map: Optional[Dict[str, str]] = None

    def __init__(
        self,
        dataset_root: Union[str, Path] = "/mnt/external_usb_hdd/6YL/Datasets/3RScan",
        *,
        meta_path: Optional[Union[str, Path]] = None, 
        meta_file: str = "meta.parquet",
        modality: list[str] = ["image", "graph"],
        rebuild_meta: bool = False,
        save_meta: bool = False,
        limit: Optional[int] = None,
        scene_list_path: Optional[Union[str, Path]] = None,
        scene_filter_mode: str = "all", # "all", "listed", "same_room_excluding_listed"
        room_json_path: Union[str, Path] = "/mnt/external_usb_hdd/6YL/Datasets/3RScan/files/3RScan.json",
        image_transform: Any = DefaultImageTransform(resize=(320, 192), train=False),
        graph_feat_dim: int = 4,
        graph_rotate: bool = True,
    ) -> None:
    
        super().__init__()
        self.dataset_root = Path(dataset_root)
        self.modality = modality
        if not self.dataset_root.exists():
            raise FileNotFoundError(f"Given dataset_root={self.dataset_root} doesn't exist")

        self.meta_path = self.dataset_root / meta_file if meta_path is None else Path(meta_path) / meta_file
        if scene_list_path is None and scene_filter_mode != "all":
            scene_list_path = self.dataset_root / "files" / "test_resplit_scans.txt"
        self.scene_list_path = Path(scene_list_path) if scene_list_path is not None else None
        self.scene_filter_mode = scene_filter_mode
        self.room_json_path = Path(room_json_path)
        selected_scene_ids = resolve_3rscan_scene_filter(
            scene_list_path=self.scene_list_path,
            scene_filter_mode=self.scene_filter_mode,
            room_json_path=self.room_json_path,
        )

        can_use_cached_meta = self.meta_path.exists() and not rebuild_meta #and selected_scene_ids is None
        if can_use_cached_meta:
            self.df = pd.read_parquet(self.meta_path)
        else:
            logger.info(
                "Rebuilding metadata for 3rscan dataset"
                if self.meta_path.exists()
                else "Metadata not found, rebuilding metadata for 3rscan dataset"
            )
            self.df = build_3rscan_df(
                self.dataset_root,
                limit=limit,
                modality=modality,
                scene_ids=selected_scene_ids,
                scene_to_room_map=self._get_scene_to_room_map(),
            )
        

        if selected_scene_ids is not None and can_use_cached_meta:
            self.df = self.df[self.df["scene"].astype(str).isin(selected_scene_ids)].reset_index(drop=True)
            self.df["idx"] = np.arange(len(self.df), dtype=np.int64)

        if limit is not None and can_use_cached_meta:
            self.df = self.df.iloc[:limit].reset_index(drop=True)
            self.df["idx"] = np.arange(len(self.df), dtype=np.int64)

        # missing cols checking
        missing_cols = {"idx", "pose", "graph_path", "scene", "image_path"} - set(self.df.columns)
        if missing_cols:
            raise ValueError(f"{self.meta_path} is missing columns: {sorted(missing_cols)}, {self.df.columns}")
        
        self.image_transform = image_transform
        self.graph_feat_dim = graph_feat_dim
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
                img = img.permute(2, 0, 1)  # HWC -> CHW
            img = img.float() / 255.0
        return img

    def _load_graph(self, graph_path: Union[str, Path]) -> Tensor:
        if graph_path is None:
            return None

        graph = torch.load(graph_path, map_location="cpu")
        graph = _sanitize_graph_obj(graph, feat_dim=self.graph_feat_dim)

        if isinstance(graph, list):
            out = []
            for g in graph:
                g = _ensure_nonempty(g, feat_dim=self.graph_feat_dim)
                if self.graph_rotate:
                    g = rotate_graph_features(g)
                out.append(g)
            return out

        graph = _ensure_nonempty(graph, feat_dim=self.graph_feat_dim)

        if self.graph_rotate:
            graph = rotate_graph_features(graph)

        return graph

    def __getitem__(self, idx: int) -> Dict[str, Any]:  
        row = self.df.iloc[int(idx)]
        scene = row["scene"]
        room = row["room"]
        pose = torch.tensor(np.asarray(row["pose"], dtype=np.float32), dtype=torch.float32)

        frame = {
            "idx": torch.tensor(int(row["idx"]), dtype=torch.int64),
            # Keep the original scene id so callers can reason about rooms/scenes.
            "scene": str(scene),
            "room": str(room),
            "scene_hash": torch.tensor(hash(str(scene))),
            "pose": pose,
        }
        if "image" in self.modality:
            frame["image_main"] = self._load_image(row["image_path"])
        if "graph" in self.modality:
            frame["graph_main"] = self._load_graph(row["graph_path"])
        return frame

    @classmethod
    def _get_scene_to_room_map(
        cls,
        *,
        room_json_path: Union[str, Path] = "/mnt/external_usb_hdd/6YL/Datasets/3RScan/files/3RScan.json",
    ) -> Dict[str, str]:
        if cls._scene_to_room_map is not None:
            return cls._scene_to_room_map

        cls._scene_to_room_map = build_scene_to_room_map(room_json_path=room_json_path)
        return cls._scene_to_room_map

    def similarity_check(
        self,
        a: Dict[str, Any],
        b: Dict[str, Any],
        *,  
        mode: str = "pose", # "pose" or "room"
        trans_tol_m: float = 3.0,
        rot_tol_deg: float = 60.0,
    ) -> bool:
        """
        Return True if:
        1) both samples belong to the same room (via `3RScan.json` grouping), and
        2) their pose is within `trans_tol_m` translation and `rot_tol_deg` rotation.

        Expected pose format is `[tx, ty, tz, qx, qy, qz, qw]`.
        """

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
        if rot_diff_deg > float(rot_tol_deg):
            return False

        return True

    
    @staticmethod
    def collate_fn(batch: Sequence[Dict[str, Tensor]]) -> Dict[str, Tensor]:
        """Collate function that keeps paths/scenes as Python lists.""" 
        out = {
            "idxs": torch.stack([b["idx"] for b in batch], dim=0),
            "poses": torch.stack([b["pose"] for b in batch], dim=0),
            "scenes_hashes": torch.stack([b["scene_hash"] for b in batch], dim=0),
        }
        if "image_main" in batch[0].keys():
            out["images_main"] = torch.stack([b["image_main"] for b in batch], dim=0)
        if "graph_main" in batch[0].keys():
            out["graphs_main"] = _collate_graph_objects([(b["graph_main"]) for b in batch], feat_dim=4)
        return out

