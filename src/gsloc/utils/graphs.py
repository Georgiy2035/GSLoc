import numpy as np
import torch
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data, HeteroData
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple


def dict_to_pyg_data(d, feat_dim, edge_attr_dim):
    if d is None:
        return None

    x = d.get("x", None)
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, dtype=torch.float32)
    elif x is not None and not torch.is_tensor(x):
        x = torch.tensor(np.asarray(x), dtype=torch.float32)

    if x is None or x.numel() == 0 or x.ndim != 2 or x.shape[0] == 0:
        x = torch.zeros((1, feat_dim), dtype=torch.float32)
    else:
        x = x.float()

    num_nodes = x.shape[0]

    edge_index = d.get("edge_index", None)
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        if not torch.is_tensor(edge_index):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        else:
            edge_index = edge_index.long()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_attr = d.get("edge_attr", None)
    if edge_attr is None:
        edge_attr = torch.empty((0, edge_attr_dim), dtype=torch.float32)
    elif not torch.is_tensor(edge_attr):
        edge_attr = torch.tensor(np.asarray(edge_attr, dtype=float), dtype=torch.float32)
    else:
        edge_attr = edge_attr.float()
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.unsqueeze(0)

    node_class = d.get("node_class", None)
    if node_class is None:
        node_class = torch.zeros((num_nodes,), dtype=torch.long)
    else:
        if not torch.is_tensor(node_class):
            node_class = torch.tensor(node_class, dtype=torch.long)
        else:
            node_class = node_class.long()
        node_class = node_class.view(-1)
        if node_class.numel() != num_nodes:
            fixed = torch.zeros((num_nodes,), dtype=torch.long)
            m = min(num_nodes, node_class.numel())
            fixed[:m] = node_class[:m]
            node_class = fixed

    edge_label = d.get("edge_label", None)
    if edge_label is not None and not torch.is_tensor(edge_label):
        edge_label = torch.tensor(edge_label, dtype=torch.long)
    if torch.is_tensor(edge_label):
        edge_label = edge_label.view(-1)

    edge_u_class = d.get("edge_u_class", None)
    if edge_u_class is not None and not torch.is_tensor(edge_u_class):
        edge_u_class = torch.tensor(edge_u_class, dtype=torch.long)
    if torch.is_tensor(edge_u_class):
        edge_u_class = edge_u_class.view(-1)

    edge_v_class = d.get("edge_v_class", None)
    if edge_v_class is not None and not torch.is_tensor(edge_v_class):
        edge_v_class = torch.tensor(edge_v_class, dtype=torch.long)
    if torch.is_tensor(edge_v_class):
        edge_v_class = edge_v_class.view(-1)

    if edge_index.numel() > 0:
        valid = (
            (edge_index[0] >= 0) & (edge_index[0] < num_nodes) &
            (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
        )
        if not valid.all():
            edge_index = edge_index[:, valid]
            edge_count = edge_index.shape[1]

            if torch.is_tensor(edge_attr):
                edge_attr = edge_attr[valid] if edge_attr.shape[0] == valid.shape[0] else edge_attr[:edge_count]
            if torch.is_tensor(edge_label):
                edge_label = edge_label[valid] if edge_label.shape[0] == valid.shape[0] else edge_label[:edge_count]
            if torch.is_tensor(edge_u_class):
                edge_u_class = edge_u_class[valid] if edge_u_class.shape[0] == valid.shape[0] else edge_u_class[:edge_count]
            if torch.is_tensor(edge_v_class):
                edge_v_class = edge_v_class[valid] if edge_v_class.shape[0] == valid.shape[0] else edge_v_class[:edge_count]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        node_class=node_class,
        edge_label=edge_label,
        edge_u_class=edge_u_class,
        edge_v_class=edge_v_class,
    )


def _ensure_nonempty(data_obj, feat_dim, feat_edge_attr_dim):
    if data_obj is None:
        return Data(
            x=torch.zeros((1, feat_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, feat_edge_attr_dim), dtype=torch.float32),
            edge_label=torch.empty((0,), dtype=torch.long),
            edge_u_class=torch.empty((0,), dtype=torch.long),
            edge_v_class=torch.empty((0,), dtype=torch.long),
            node_class=torch.zeros((1,), dtype=torch.long),
        )

    if isinstance(data_obj, dict):
        data_obj = dict_to_pyg_data(data_obj, feat_dim, feat_edge_attr_dim)

    if not isinstance(data_obj, (Data, HeteroData)):
        return data_obj

    x = getattr(data_obj, "x", None)
    if x is None or not torch.is_tensor(x):
        x = torch.as_tensor(x, dtype=torch.float32) if x is not None else None

    synthetic_node = False
    if x is None or x.numel() == 0 or x.shape[0] == 0:
        x = torch.zeros((1, feat_dim), dtype=torch.float32)
        synthetic_node = True
    else:
        x = x.float()

    n_nodes = x.shape[0]

    node_class = getattr(data_obj, "node_class", None)
    if node_class is None:
        node_class = torch.zeros((n_nodes,), dtype=torch.long)
    else:
        if not torch.is_tensor(node_class):
            node_class = torch.tensor(node_class, dtype=torch.long)
        else:
            node_class = node_class.long()
        node_class = node_class.view(-1)
        if node_class.numel() != n_nodes:
            fixed = torch.zeros((n_nodes,), dtype=torch.long)
            m = min(n_nodes, node_class.numel())
            fixed[:m] = node_class[:m]
            node_class = fixed

    if synthetic_node:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, feat_edge_attr_dim), dtype=torch.float32)
        edge_label = torch.empty((0,), dtype=torch.long)
        edge_u_class = torch.empty((0,), dtype=torch.long)
        edge_v_class = torch.empty((0,), dtype=torch.long)
    else:
        edge_index = getattr(data_obj, "edge_index", None)
        if edge_index is None:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        else:
            if not torch.is_tensor(edge_index):
                edge_index = torch.tensor(edge_index, dtype=torch.long)
            else:
                edge_index = edge_index.long()
            if edge_index.ndim != 2 or edge_index.shape[0] != 2:
                edge_index = torch.empty((2, 0), dtype=torch.long)

        edge_attr = getattr(data_obj, "edge_attr", None)
        if edge_attr is None:
            edge_attr = torch.empty((0, feat_edge_attr_dim), dtype=torch.float32)
        elif not torch.is_tensor(edge_attr):
            edge_attr = torch.tensor(np.asarray(edge_attr, dtype=float), dtype=torch.float32)
        else:
            edge_attr = edge_attr.float()

        edge_label = getattr(data_obj, "edge_label", None)
        if edge_label is None:
            edge_label = torch.empty((0,), dtype=torch.long)
        elif not torch.is_tensor(edge_label):
            edge_label = torch.tensor(edge_label, dtype=torch.long)
        else:
            edge_label = edge_label.long()
        edge_label = edge_label.view(-1)

        edge_u_class = getattr(data_obj, "edge_u_class", None)
        if edge_u_class is None:
            edge_u_class = torch.empty((0,), dtype=torch.long)
        elif not torch.is_tensor(edge_u_class):
            edge_u_class = torch.tensor(edge_u_class, dtype=torch.long)
        else:
            edge_u_class = edge_u_class.long()
        edge_u_class = edge_u_class.view(-1)

        edge_v_class = getattr(data_obj, "edge_v_class", None)
        if edge_v_class is None:
            edge_v_class = torch.empty((0,), dtype=torch.long)
        elif not torch.is_tensor(edge_v_class):
            edge_v_class = torch.tensor(edge_v_class, dtype=torch.long)
        else:
            edge_v_class = edge_v_class.long()
        edge_v_class = edge_v_class.view(-1)

        if edge_index.numel() > 0:
            valid = (
                (edge_index[0] >= 0) & (edge_index[0] < n_nodes) &
                (edge_index[1] >= 0) & (edge_index[1] < n_nodes)
            )
            if not valid.all():
                edge_index = edge_index[:, valid]
                edge_count = edge_index.shape[1]
                if edge_attr.shape[0] == valid.shape[0]:
                    edge_attr = edge_attr[valid]
                else:
                    edge_attr = edge_attr[:edge_count]
                if edge_label.shape[0] == valid.shape[0]:
                    edge_label = edge_label[valid]
                else:
                    edge_label = edge_label[:edge_count]
                if edge_u_class.shape[0] == valid.shape[0]:
                    edge_u_class = edge_u_class[valid]
                else:
                    edge_u_class = edge_u_class[:edge_count]
                if edge_v_class.shape[0] == valid.shape[0]:
                    edge_v_class = edge_v_class[valid]
                else:
                    edge_v_class = edge_v_class[:edge_count]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        edge_u_class=edge_u_class,
        edge_v_class=edge_v_class,
        node_class=node_class,
    )

def _sanitize_graph_obj(g, feat_dim, feat_edge_attr_dim):
    if g is None:
        return None

    if isinstance(g, PyGBatch):
        # batch -> list[Data]
        return [x for x in g.to_data_list()]

    if isinstance(g, dict):
        g = dict_to_pyg_data(g, feat_dim, feat_edge_attr_dim)

    if isinstance(g, list):
        out = []
        for x in g:
            sx = _sanitize_graph_obj(x, feat_dim, feat_edge_attr_dim)
            if sx is None:
                continue
            if isinstance(sx, list):
                out.extend(sx)
            else:
                out.append(sx)
        return out

    if not isinstance(g, (Data, HeteroData)):
        return g

    # normalize tensors
    x = getattr(g, "x", None)
    if x is None:
        x = torch.zeros((1, feat_dim), dtype=torch.float32)
    elif not torch.is_tensor(x):
        x = torch.tensor(x, dtype=torch.float32)
    else:
        x = x.float()
    if x.ndim != 2 or x.shape[0] == 0:
        x = torch.zeros((1, feat_dim), dtype=torch.float32)

    num_nodes = x.shape[0]

    node_class = getattr(g, "node_class", None)
    if node_class is None:
        node_class = torch.zeros((num_nodes,), dtype=torch.long)
    else:
        if not torch.is_tensor(node_class):
            node_class = torch.tensor(node_class, dtype=torch.long)
        else:
            node_class = node_class.long()
        node_class = node_class.view(-1)
        if node_class.numel() != num_nodes:
            fixed = torch.zeros((num_nodes,), dtype=torch.long)
            m = min(num_nodes, node_class.numel())
            fixed[:m] = node_class[:m]
            node_class = fixed

    edge_index = getattr(g, "edge_index", None)
    if edge_index is None:
        edge_index = torch.empty((2, 0), dtype=torch.long)
    else:
        if not torch.is_tensor(edge_index):
            edge_index = torch.tensor(edge_index, dtype=torch.long)
        else:
            edge_index = edge_index.long()
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            edge_index = torch.empty((2, 0), dtype=torch.long)

    edge_attr = getattr(g, "edge_attr", None)
    if edge_attr is None:
        edge_attr = torch.empty((0, feat_edge_attr_dim), dtype=torch.float32)
    elif not torch.is_tensor(edge_attr):
        edge_attr = torch.tensor(np.asarray(edge_attr, dtype=float), dtype=torch.float32)
    else:
        edge_attr = edge_attr.float()
    if edge_attr.ndim == 1:
        edge_attr = edge_attr.unsqueeze(0)

    edge_label = getattr(g, "edge_label", None)
    if edge_label is None:
        edge_label = torch.empty((0,), dtype=torch.long)
    elif not torch.is_tensor(edge_label):
        edge_label = torch.tensor(edge_label, dtype=torch.long)
    else:
        edge_label = edge_label.long()
    edge_label = edge_label.view(-1)

    edge_u_class = getattr(g, "edge_u_class", None)
    if edge_u_class is None:
        edge_u_class = torch.empty((0,), dtype=torch.long)
    elif not torch.is_tensor(edge_u_class):
        edge_u_class = torch.tensor(edge_u_class, dtype=torch.long)
    else:
        edge_u_class = edge_u_class.long()
    edge_u_class = edge_u_class.view(-1)

    edge_v_class = getattr(g, "edge_v_class", None)
    if edge_v_class is None:
        edge_v_class = torch.empty((0,), dtype=torch.long)
    elif not torch.is_tensor(edge_v_class):
        edge_v_class = torch.tensor(edge_v_class, dtype=torch.long)
    else:
        edge_v_class = edge_v_class.long()
    edge_v_class = edge_v_class.view(-1)

    if edge_index.numel() > 0:
        valid = (
            (edge_index[0] >= 0) & (edge_index[0] < num_nodes) &
            (edge_index[1] >= 0) & (edge_index[1] < num_nodes)
        )
        if not valid.all():
            edge_index = edge_index[:, valid]
            edge_count = edge_index.shape[1]
            if edge_attr.shape[0] == valid.shape[0]:
                edge_attr = edge_attr[valid]
            else:
                edge_attr = edge_attr[:edge_count]
            if edge_label.shape[0] == valid.shape[0]:
                edge_label = edge_label[valid]
            else:
                edge_label = edge_label[:edge_count]
            if edge_u_class.shape[0] == valid.shape[0]:
                edge_u_class = edge_u_class[valid]
            else:
                edge_u_class = edge_u_class[:edge_count]
            if edge_v_class.shape[0] == valid.shape[0]:
                edge_v_class = edge_v_class[valid]
            else:
                edge_v_class = edge_v_class[:edge_count]

    return Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        edge_label=edge_label,
        edge_u_class=edge_u_class,
        edge_v_class=edge_v_class,
        node_class=node_class,
    )

def rotate_graph_features(graph):
        """
        Поворот graph['x'] на 90° по часовой стрелке.
        [1 - y, x, h, w]
        """
        if graph is None or not hasattr(graph, "x") or graph.x is None:
            return graph

        x = graph.x
        if not torch.is_tensor(x):
            x = torch.as_tensor(x, dtype=torch.float32)
        else:
            x = x.float()

        if x.ndim != 2 or x.shape[1] < 2:
            graph.x = x
            return graph

        new_x = x.clone()

        if x.shape[1] >= 4:
            new_x[:, 0] = 1.0 - x[:, 1]
            new_x[:, 1] = x[:, 0]
            new_x[:, 2] = x[:, 3]
            new_x[:, 3] = x[:, 2]
        else:
            new_x[:, 0] = 1.0 - x[:, 1]
            new_x[:, 1] = x[:, 0]

        graph.x = new_x
        return graph


def _graph_to_list(g):
    if g is None:
        return []
    if isinstance(g, (list, tuple)):
        return [x for x in g if x is not None]
    return [g]


def _ensure_graph_list(graphs, feat_dim, feat_edge_attr_dim=7):
    flat = []
    for g in _graph_to_list(graphs):
        sg = _sanitize_graph_obj(g, feat_dim, feat_edge_attr_dim)
        if sg is None:
            continue
        if isinstance(sg, list):
            for x in sg:
                x = _ensure_nonempty(_sanitize_graph_obj(x, feat_dim, feat_edge_attr_dim), feat_dim, feat_edge_attr_dim)
                flat.append(x)
        else:
            flat.append(_ensure_nonempty(sg, feat_dim, feat_edge_attr_dim))
    return flat


def _collate_graph_objects(graphs, feat_dim, feat_edge_attr_dim=7):
    graphs = _ensure_graph_list(graphs, feat_dim=feat_dim, feat_edge_attr_dim=feat_edge_attr_dim)
    if len(graphs) == 0:
        return None
    if isinstance(graphs[0], (Data, HeteroData)):
        return PyGBatch.from_data_list(graphs)
    if torch.is_tensor(graphs[0]):
        if all(torch.is_tensor(g) and g.shape == graphs[0].shape for g in graphs):
            return torch.stack(graphs, dim=0)
    return graphs


DEFAULT_EDGE_LABEL2IDX: Dict[str, int] = {
    "none": 0,
    "supported by": 1,
    "left": 2,
    "right": 3,
    "front": 4,
    "behind": 5,
    "close by": 6,
    "inside": 7,
    "bigger than": 8,
    "smaller than": 9,
    "higher than": 10,
    "lower than": 11,
    "same symmetry as": 12,
    "same as": 13,
    "attached to": 14,
    "standing on": 15,
    "lying on": 16,
    "hanging on": 17,
    "connected to": 18,
    "leaning against": 19,
    "part of": 20,
    "belonging to": 21,
    "build in": 22,
    "standing in": 23,
    "cover": 24,
    "lying in": 25,
    "hanging in": 26,
    "same color": 27,
    "same material": 28,
    "same texture": 29,
    "same shape": 30,
    "same state": 31,
    "same object type": 32,
    "messier than": 33,
    "cleaner than": 34,
    "fuller than": 35,
    "more closed": 36,
    "more open": 37,
    "brighter than": 38,
    "darker than": 39,
    "more comfortable than": 40,
}


def _rotate_xyxy_clockwise_90(xyxy: List[float]) -> List[float]:
    x1, y1, x2, y2 = [float(v) for v in xyxy]
    return [1.0 - y2, x1, 1.0 - y1, x2]


def _safe_bbox_from_node(node_data: Dict[str, Any]) -> Tuple[List[float], List[float], float, float]:
    bbox = node_data.get("bbox_2d", {}) if isinstance(node_data, dict) else {}
    xyxy_raw = bbox.get("xyxy", None)
    if not isinstance(xyxy_raw, (list, tuple)) or len(xyxy_raw) != 4:
        xyxy_raw = [0.0, 0.0, 1.0, 1.0]
    xyxy_raw = [float(v) for v in xyxy_raw]

    xyxy_rot = _rotate_xyxy_clockwise_90(xyxy_raw)
    rx1, ry1, rx2, ry2 = xyxy_rot
    rw = max(float(rx2 - rx1), 1e-6)
    rh = max(float(ry2 - ry1), 1e-6)
    return xyxy_raw, xyxy_rot, rw, rh


def _bbox_iou_and_intersection_ratio(source_xyxy: List[float], target_xyxy: List[float]) -> Tuple[float, float]:
    sx1, sy1, sx2, sy2 = source_xyxy
    tx1, ty1, tx2, ty2 = target_xyxy

    ix1 = max(sx1, tx1)
    iy1 = max(sy1, ty1)
    ix2 = min(sx2, tx2)
    iy2 = min(sy2, ty2)
    iw = max(ix2 - ix1, 0.0)
    ih = max(iy2 - iy1, 0.0)
    inter = iw * ih

    s_area = max((sx2 - sx1) * (sy2 - sy1), 1e-9)
    t_area = max((tx2 - tx1) * (ty2 - ty1), 1e-9)
    union = max(s_area + t_area - inter, 1e-9)

    iou = float(inter / union)
    inter_over_source = float(inter / s_area)
    return iou, inter_over_source


def convert_scenegraph_json_to_compact_pt(
    json_path: Path,
    *,
    edge_label2idx: Dict[str, int] | None = None,
    graph_rotated: bool = True,
) -> Dict[str, Any]:
    if edge_label2idx is None:
        edge_label2idx = dict(DEFAULT_EDGE_LABEL2IDX)

    with Path(json_path).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    nodes = raw.get("nodes", [])
    links = raw.get("links", [])

    node_id_to_idx: Dict[int, int] = {}
    node_feats: List[List[float]] = []
    node_class: List[int] = []
    node_meta: List[Dict[str, Any]] = []
    node_bboxes_rot: List[List[float]] = []
    node_wh_rot: List[Tuple[float, float]] = []

    for idx, node in enumerate(nodes):
        node_id = int(node.get("id", idx))
        data = node.get("data", {}) if isinstance(node, dict) else {}

        node_id_to_idx[node_id] = idx

        xyxy_raw, xyxy_rot, rw, rh = _safe_bbox_from_node(data)
        rx1, ry1, rx2, ry2 = xyxy_rot
        cx_rot = 0.5 * (rx1 + rx2)
        cy_rot = 0.5 * (ry1 + ry2)

        node_feats.append([float(cx_rot), float(cy_rot), float(rw), float(rh)])
        node_class.append(int(data.get("class_id", 0)))
        node_bboxes_rot.append(xyxy_rot)
        node_wh_rot.append((rw, rh))
        node_meta.append(
            {
                "id": node_id,
                "class_name": str(data.get("class_name", "unknown")),
                "xyxy_raw": xyxy_raw,
                "xyxy_rot": xyxy_rot,
                "center_rot": [float(cx_rot), float(cy_rot)],
                "wh_rot": [float(rw), float(rh)],
            }
        )

    edge_index_src: List[int] = []
    edge_index_dst: List[int] = []
    edge_attr: List[List[float]] = []
    edge_label: List[int] = []
    edge_u_class: List[int] = []
    edge_v_class: List[int] = []
    edge_meta: List[Dict[str, Any]] = []

    eps = 1e-9
    for edge in links:
        src_id = int(edge.get("source", -1))
        dst_id = int(edge.get("target", -1))
        if src_id not in node_id_to_idx or dst_id not in node_id_to_idx:
            continue

        u = node_id_to_idx[src_id]
        v = node_id_to_idx[dst_id]

        ux, uy, uw, uh = node_feats[u]
        vx, vy, vw, vh = node_feats[v]

        dx = float(vx - ux)
        dy = float(vy - uy)
        dist = float(math.sqrt(dx * dx + dy * dy))
        inv_dist = 1.0 / max(dist, eps)

        iou, inter_over_source = _bbox_iou_and_intersection_ratio(
            node_bboxes_rot[u],
            node_bboxes_rot[v],
        )

        src_area = max(node_wh_rot[u][0] * node_wh_rot[u][1], eps)
        dst_area = max(node_wh_rot[v][0] * node_wh_rot[v][1], eps)
        area_ratio = float(dst_area / src_area)
        width_ratio = float(vw / max(uw, eps))
        height_ratio = float(vh / max(uh, eps))

        edge_index_src.append(u)
        edge_index_dst.append(v)
        edge_attr.append(
            [
                dist,
                dx,
                dy,
                float(dy * inv_dist),
                float(dx * inv_dist),
                float(iou),
                float(inter_over_source),
                float(math.log1p(area_ratio)),
                float(math.log1p(width_ratio)),
                float(math.log1p(height_ratio)),
            ]
        )

        rel = str(edge.get("label", "none")).strip().lower()
        if rel not in edge_label2idx:
            edge_label2idx[rel] = max(edge_label2idx.values(), default=0) + 1
        edge_label.append(int(edge_label2idx[rel]))

        edge_u_class.append(int(node_class[u]))
        edge_v_class.append(int(node_class[v]))
        edge_meta.append({"u": int(src_id), "v": int(dst_id), "label": rel})

    n_nodes = len(node_feats)
    n_edges = len(edge_attr)

    out = {
        "x": torch.tensor(node_feats, dtype=torch.float32) if n_nodes > 0 else torch.empty((0, 4), dtype=torch.float32),
        "edge_index": (
            torch.tensor([edge_index_src, edge_index_dst], dtype=torch.long)
            if n_edges > 0
            else torch.empty((2, 0), dtype=torch.long)
        ),
        "edge_attr": torch.tensor(edge_attr, dtype=torch.float32) if n_edges > 0 else torch.empty((0, 10), dtype=torch.float32),
        "node_class": torch.tensor(node_class, dtype=torch.long) if n_nodes > 0 else torch.empty((0,), dtype=torch.long),
        "edge_label": torch.tensor(edge_label, dtype=torch.long) if n_edges > 0 else torch.empty((0,), dtype=torch.long),
        "edge_u_class": torch.tensor(edge_u_class, dtype=torch.long) if n_edges > 0 else torch.empty((0,), dtype=torch.long),
        "edge_v_class": torch.tensor(edge_v_class, dtype=torch.long) if n_edges > 0 else torch.empty((0,), dtype=torch.long),
        "node_meta": node_meta,
        "edge_meta": edge_meta,
        "edge_label2idx": dict(edge_label2idx),
        "json_path": str(json_path),
        "scan_id": json_path.parent.name,
        "graph_rotated": bool(graph_rotated),
    }
    return out


def convert_all_scenegraph_jsons_to_pt(
    src_root: Path | str = "/mnt/external_usb_hdd/6YL/Datasets/3RScan/SceneGraph_makarov",
    dst_root: Path | str = "/mnt/external_usb_hdd/6YL/Datasets/3RScan/SceneGraph_makarov_pt_compact",
    *,
    overwrite: bool = False,
    edge_label2idx: Dict[str, int] | None = None,
) -> Dict[str, int]:
    """
    Convert all 3RScan scene-graph JSON files to compact `.pt` graph files.

    Expected source layout:
      src_root/<scene_id>/frame-XXXXXX.json
    Saved layout:
      dst_root/<scene_id>/frame-XXXXXX.pt
    """
    src_root = Path(src_root)
    dst_root = Path(dst_root)
    if not src_root.exists():
        raise FileNotFoundError(f"Source graph directory does not exist: {src_root}")

    edge_label2idx_local = dict(DEFAULT_EDGE_LABEL2IDX if edge_label2idx is None else edge_label2idx)

    converted = 0
    skipped_existing = 0
    failed = 0

    for scene_dir in sorted(p for p in src_root.iterdir() if p.is_dir()):
        out_scene_dir = dst_root / scene_dir.name
        out_scene_dir.mkdir(parents=True, exist_ok=True)

        for json_path in sorted(scene_dir.glob("*.json")):
            out_path = out_scene_dir / f"{json_path.stem}.pt"
            if out_path.exists() and not overwrite:
                skipped_existing += 1
                continue
            try:
                graph_dict = convert_scenegraph_json_to_compact_pt(
                    json_path,
                    edge_label2idx=edge_label2idx_local,
                    graph_rotated=True,
                )
                torch.save(graph_dict, out_path)
                converted += 1
            except Exception:
                failed += 1

    return {
        "converted": converted,
        "skipped_existing": skipped_existing,
        "failed": failed,
        "num_edge_labels": len(edge_label2idx_local),
        "src_root": str(src_root),
        "dst_root": str(dst_root),
    }