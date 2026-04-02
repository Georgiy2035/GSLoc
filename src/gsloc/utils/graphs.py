import numpy as np
import torch
from torch_geometric.data import Batch as PyGBatch
from torch_geometric.data import Data, HeteroData


def dict_to_pyg_data(d, feat_dim=4, edge_attr_dim=7):
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


def _ensure_nonempty(data_obj, feat_dim=4):
    if data_obj is None:
        return Data(
            x=torch.zeros((1, feat_dim), dtype=torch.float32),
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_attr=torch.empty((0, 7), dtype=torch.float32),
            edge_label=torch.empty((0,), dtype=torch.long),
            edge_u_class=torch.empty((0,), dtype=torch.long),
            edge_v_class=torch.empty((0,), dtype=torch.long),
            node_class=torch.zeros((1,), dtype=torch.long),
        )

    if isinstance(data_obj, dict):
        data_obj = dict_to_pyg_data(data_obj, feat_dim=feat_dim)

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
        edge_attr = torch.empty((0, 7), dtype=torch.float32)
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
            edge_attr = torch.empty((0, 7), dtype=torch.float32)
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



def _sanitize_graph_obj(g, feat_dim=4):
    if g is None:
        return None

    if isinstance(g, PyGBatch):
        # batch -> list[Data]
        return [x for x in g.to_data_list()]

    if isinstance(g, dict):
        g = dict_to_pyg_data(g, feat_dim=feat_dim)

    if isinstance(g, list):
        out = []
        for x in g:
            sx = _sanitize_graph_obj(x, feat_dim=feat_dim)
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
        edge_attr = torch.empty((0, 7), dtype=torch.float32)
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


def _ensure_graph_list(graphs, feat_dim=4):
    flat = []
    for g in _graph_to_list(graphs):
        sg = _sanitize_graph_obj(g, feat_dim=feat_dim)
        if sg is None:
            continue
        if isinstance(sg, list):
            for x in sg:
                x = _ensure_nonempty(_sanitize_graph_obj(x, feat_dim=feat_dim), feat_dim=feat_dim)
                flat.append(x)
        else:
            flat.append(_ensure_nonempty(sg, feat_dim=feat_dim))
    return flat


def _collate_graph_objects(graphs, feat_dim=4):
    graphs = _ensure_graph_list(graphs, feat_dim=feat_dim)
    if len(graphs) == 0:
        return None
    if isinstance(graphs[0], (Data, HeteroData)):
        return PyGBatch.from_data_list(graphs)
    if torch.is_tensor(graphs[0]):
        if all(torch.is_tensor(g) and g.shape == graphs[0].shape for g in graphs):
            return torch.stack(graphs, dim=0)
    return graphs