import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.nn import GINEConv, global_mean_pool, global_max_pool


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
                 n_layers=2,
                 proj_dim=64,
                 num_node_classes=None,
                 node_emb_dim=16,
                 num_edge_classes=None,
                 edge_emb_dim=16,
                 dropout=0):
        super().__init__()

        self.use_node_class = (num_node_classes is not None)
        self.use_edge_label = (num_edge_classes is not None)

        self.node_emb = None
        if self.use_node_class:
            self.node_emb = nn.Embedding(num_node_classes, node_emb_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)

        self.edge_emb = None
        self.edge_proj = None
        if self.use_edge_label:
            self.edge_emb = nn.Embedding(num_edge_classes, edge_emb_dim)
            nn.init.xavier_uniform_(self.edge_emb.weight)
            self.edge_proj = nn.Sequential(
                nn.Linear(edge_emb_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

        eff_in_dim = in_dim + (node_emb_dim if self.use_node_class else 0)

        self.input_mlp = nn.Sequential(
            nn.Linear(eff_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.convs = nn.ModuleList()
        for _ in range(n_layers):
            mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(GINEConv(nn=mlp, train_eps=True))

        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)

        self.pool_out_dim = hidden_dim * 2
        self.proj = nn.Sequential(
            nn.Linear(self.pool_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        self._proj_dim = proj_dim

    def forward(self, batch):
        batch = batch["graphs_main"]
        x = batch.x

        #print("batch in forward", batch)
        if self.use_node_class and hasattr(batch, 'node_class') and batch.node_class is not None:
            node_cls = batch.node_class.long().to(x.device)
            node_emb = self.node_emb(node_cls)
            x = torch.cat([x, node_emb], dim=1)

        h = self.input_mlp(x)

        edge_attr = None
        if self.use_edge_label and hasattr(batch, 'edge_label') and batch.edge_label is not None:
            edge_label = batch.edge_label.long().to(x.device)
            edge_attr = self.edge_emb(edge_label)
            edge_attr = self.edge_proj(edge_attr)

        for conv in self.convs:
            h = conv(h, batch.edge_index, edge_attr)
            h = self.act(h)
            h = self.drop(h)

        hg_mean = global_mean_pool(h, batch.batch)
        hg_max = global_max_pool(h, batch.batch)
        hg = torch.cat([hg_mean, hg_max], dim=1)

        z = self.proj(hg)
        z = F.normalize(z, p=2, dim=1)
        output = {"final_descriptor": z}
        return output

    @property
    def out_dim(self):
        return self._proj_dim

class MultiModalVPRGraphEncoder(nn.Module):
    """Fuses a frozen MegaLoc-style image tower (8448-D) with ``VPRGraphEncoder``.

    Module layout matches checkpoints such as ``best_model.pth`` (``graph_encoder.*``,
    ``graph_gate``, ``graph_proj``, ``image_proj``, ``fuse_norm``, ``fuse_mlp``).
    Image-backbone weights are not stored in that checkpoint; only ``image_encoder`` hub
    weights are used unless you load them separately.
    """

    def __init__(
        self,
        graph_encoder: nn.Module,
        image_encoder: nn.Module,
        *,
        image_out_dim: int = 8448,
        fusion_dim: int = 8448,
        graph_fusion_scale: float = 0.05,
        fuse_mlp_residual_scale: float = 0.1,
        normalize: bool = True,
        mode: str = "fusion",
        freeze_image_encoder: bool = True,
    ):
        super().__init__()
        self.graph_encoder = graph_encoder
        self.image_encoder = image_encoder
        self.normalize = normalize
        self.mode = mode
        self.graph_fusion_scale = graph_fusion_scale
        self.fuse_mlp_residual_scale = fuse_mlp_residual_scale

        graph_dim = int(getattr(graph_encoder, "out_dim"))
        if graph_dim <= 0:
            raise ValueError("graph_encoder must expose a positive out_dim")

        self.graph_proj = nn.Sequential(
            nn.Linear(graph_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
        )
        self.graph_gate = nn.Sequential(nn.Linear(graph_dim, fusion_dim))
        self.image_proj = nn.Sequential(
            nn.Linear(image_out_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
        )
        self.fuse_norm = nn.LayerNorm(fusion_dim)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
        )

        if freeze_image_encoder:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self._fusion_dim = fusion_dim
        self._graph_dim = graph_dim

    @property
    def out_dim(self):
        return self._graph_dim if self.mode == "graph" else self._fusion_dim

    def freeze_graph(self):
        for p in self.graph_encoder.parameters():
            p.requires_grad = False

    def unfreeze_graph(self):
        for p in self.graph_encoder.parameters():
            p.requires_grad = True

    def freeze_image(self):
        for p in self.image_encoder.parameters():
            p.requires_grad = False

    def unfreeze_image(self):
        for p in self.image_encoder.parameters():
            p.requires_grad = True

    def forward(self, batch: dict, return_parts: bool = False):
        del return_parts  # API compatibility; always return a dict for pipelines.
        out: dict = {}
        graph = batch.get("graphs_main")
        image = batch.get("images_main")

        if self.mode == "graph":
            if graph is None:
                raise ValueError("mode='graph' requires batch['graphs_main']")
            g_out = self.graph_encoder(batch)
            z = _extract_embedding(g_out)
            out["graph"] = z
            out["final_descriptor"] = z
            return out

        if self.mode == "image":
            if image is None:
                raise ValueError("mode='image' requires batch['images_main']")
            raw = self.image_encoder(batch)
            z_img = _extract_embedding(raw)
            feat = self.image_proj(z_img)
            if self.normalize:
                feat = F.normalize(feat, p=2, dim=1)
            out["image"] = feat
            out["final_descriptor"] = feat
            return out

        if self.mode == "fusion":
            if graph is None or image is None:
                raise ValueError("mode='fusion' requires batch['graphs_main'] and batch['images_main']")
            graph_z = _extract_embedding(self.graph_encoder(batch))
            image_raw = _extract_embedding(self.image_encoder(batch))
            graph_feat = self.graph_proj(graph_z)
            gate = self.graph_gate(graph_z)
            image_feat = self.image_proj(image_raw)
            fused = image_feat + self.graph_fusion_scale * gate * graph_feat
            fused = self.fuse_norm(fused)
            fused = fused + self.fuse_mlp_residual_scale * self.fuse_mlp(fused)
            if self.normalize:
                fused = F.normalize(fused, p=2, dim=1)
            out["graph"] = graph_z
            out["image"] = image_feat
            out["final_descriptor"] = fused
            return out

        raise ValueError(f"Unknown mode: {self.mode}")
