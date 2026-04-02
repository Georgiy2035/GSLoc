import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool


def _extract_embedding(x):
    """
    Универсально вытаскивает embedding из:
      - Tensor
      - tuple/list
      - dict
    """
    if x is None:
        return None

    if torch.is_tensor(x):
        return x

    if isinstance(x, (tuple, list)):
        return _extract_embedding(x[0])

    if isinstance(x, dict):
        for key in ["embedding", "feat", "features", "output", "out", "z"]:
            if key in x:
                return _extract_embedding(x[key])
        # fallback: первый элемент dict
        return _extract_embedding(next(iter(x.values())))

    raise TypeError(f"Unsupported encoder output type: {type(x)}")


class VPRGraphEncoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim=256,
                 n_layers=3,
                 proj_dim=128,
                 num_node_classes=None,
                 node_emb_dim=64,
                 dropout=0.1):
        super().__init__()
        self.use_node_class = (num_node_classes is not None)
        self.node_emb = None
        if self.use_node_class:
            self.node_emb = nn.Embedding(num_node_classes, node_emb_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)
        eff_in_dim = in_dim + (node_emb_dim if self.use_node_class else 0)
        self.input_mlp = nn.Sequential(
            nn.Linear(eff_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )
        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)
        self.pool_out_dim = hidden_dim * 2
        self.proj = nn.Sequential(
            nn.Linear(self.pool_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        self._proj_dim = proj_dim

    def forward(self, batch: Batch):
        graph = batch["graphs_main"]
        x = graph.x
        if self.use_node_class and hasattr(graph, 'node_class') and graph.node_class is not None:
            node_cls = graph.node_class.long().to(x.device)
            emb = self.node_emb(node_cls)
            x = torch.cat([x, emb], dim=1)
        h = self.input_mlp(x)
        for conv in self.convs:
            h = conv(h, graph.edge_index)
            h = self.act(h)
            h = self.drop(h)
        hg_mean = global_mean_pool(h, graph.batch)
        hg_max  = global_max_pool(h, graph.batch)
        hg = torch.cat([hg_mean, hg_max], dim=1)
        z = self.proj(hg)
        z = F.normalize(z, p=2, dim=1)
        output = {"final_descriptor": z}
        return output

    @property
    def out_dim(self):
        return self._proj_dim


class MultiModalVPRGraphEncoder(nn.Module):
    def __init__(
        self,
        graph_encoder: nn.Module,
        image_encoder: nn.Module = None,
        image_out_dim: int = None,
        shared_dim: int = 128,
        fusion_hidden_dim: int = 256,
        normalize: bool = True,
    ):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.image_encoder = image_encoder
        self.normalize = normalize

        self.graph_out_dim = getattr(graph_encoder, "out_dim", None)
        if self.graph_out_dim is None:
            raise ValueError("graph_encoder must expose .out_dim")

        self.image_out_dim = image_out_dim
        self.shared_dim = shared_dim

        # graph -> shared_dim
        self.graph_proj = nn.Identity() if self.graph_out_dim == shared_dim else nn.Linear(self.graph_out_dim, shared_dim)

        # image -> shared_dim
        if self.image_encoder is not None:
            if self.image_out_dim is None:
                raise ValueError(
                    "For image_encoder you must provide image_out_dim, "
                    "unless you know the exact output dim beforehand."
                )
            self.image_proj = nn.Identity() if self.image_out_dim == shared_dim else nn.Linear(self.image_out_dim, shared_dim)
        else:
            self.image_proj = None

        # fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(shared_dim * 2, fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_hidden_dim, shared_dim),
        )

        self._out_dim = shared_dim

    @property
    def out_dim(self):
        return self._out_dim

    def freeze_graph(self):
        for p in self.graph_encoder.parameters():
            p.requires_grad = False

    def unfreeze_graph(self):
        for p in self.graph_encoder.parameters():
            p.requires_grad = True

    def freeze_image(self):
        if self.image_encoder is None:
            return
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def unfreeze_image(self):
        if self.image_encoder is None:
            return
        for p in self.image_encoder.parameters():
            p.requires_grad = True

    def unfreeze_image_last_n_blocks(self, n=1):
        """
        Простой generic helper для megaloc-like моделей.
        Если у модели есть child modules, размораживает последние n блоков.
        """
        if self.image_encoder is None:
            return

        children = list(self.image_encoder.children())
        if len(children) == 0:
            for p in self.image_encoder.parameters():
                p.requires_grad = True
            return

        for p in self.image_encoder.parameters():
            p.requires_grad = False

        for block in children[-n:]:
            for p in block.parameters():
                p.requires_grad = True

    def encode_graph(self, graph_batch):
        z = self.graph_encoder(graph_batch)   # твой VPRGraphEncoder
        z = _extract_embedding(z)
        if z.ndim > 2:
            z = torch.flatten(z, start_dim=1)
        z = self.graph_proj(z)
        if self.normalize:
            z = F.normalize(z, p=2, dim=1)
        return z

    def encode_image(self, image_batch):
        if self.image_encoder is None:
            return None
        z = self.image_encoder(image_batch)
        z = _extract_embedding(z)
        if z.ndim > 2:
            z = torch.flatten(z, start_dim=1)
        z = self.image_proj(z)
        if self.normalize:
            z = F.normalize(z, p=2, dim=1)
        return z

    def forward(
        self,
        graph=None,
        image=None,
        mode="fusion",   # "graph", "image", "fusion"
        return_parts=True,
    ):
        """
        mode:
          - graph  -> only graph embedding
          - image  -> only image embedding
          - fusion -> concat(graph, image) -> fusion head
        """
        out = {}

        graph_emb = None
        image_emb = None

        if graph is not None:
            graph_emb = self.encode_graph(graph)
            out["graph"] = graph_emb

        if image is not None:
            image_emb = self.encode_image(image)
            out["image"] = image_emb

        if mode == "graph":
            if graph_emb is None:
                raise ValueError("mode='graph' but graph is None")
            out["fused"] = graph_emb

        elif mode == "image":
            if image_emb is None:
                raise ValueError("mode='image' but image is None or image_encoder is missing")
            out["fused"] = image_emb

        elif mode == "fusion":
            if graph_emb is not None and image_emb is not None:
                fused = torch.cat([graph_emb, image_emb], dim=1)
                fused = self.fusion_head(fused)
                if self.normalize:
                    fused = F.normalize(fused, p=2, dim=1)
                out["fused"] = fused
            elif graph_emb is not None:
                out["fused"] = graph_emb
            elif image_emb is not None:
                out["fused"] = image_emb
            else:
                raise ValueError("No input modality was provided")

        else:
            raise ValueError(f"Unknown mode: {mode}")

        return out if return_parts else out["fused"]