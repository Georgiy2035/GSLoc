import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.nn import GATv2Conv, GCNConv, GINEConv, global_mean_pool, global_max_pool


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


def _sanitize_graph_edges(edge_index: torch.Tensor, num_nodes: int, edge_attr: torch.Tensor | None = None):
    """Drop invalid edges to avoid CUDA index-out-of-bounds in PyG convs."""
    if edge_index is None:
        return edge_index, edge_attr
    if edge_index.numel() == 0:
        return edge_index, edge_attr
    if edge_index.dim() != 2:
        return edge_index, None
    if edge_index.size(0) != 2 and edge_index.size(1) == 2:
        edge_index = edge_index.t()
    if edge_index.size(0) != 2:
        return edge_index, None
    edge_index = edge_index.long()
    src = edge_index[0]
    dst = edge_index[1]
    valid = (src >= 0) & (dst >= 0) & (src < num_nodes) & (dst < num_nodes)
    if bool(valid.all()):
        if edge_attr is not None and edge_attr.size(0) != edge_index.size(1):
            edge_attr = None
        return edge_index, edge_attr
    edge_index = edge_index[:, valid]
    if edge_attr is not None and edge_attr.size(0) == valid.numel():
        edge_attr = edge_attr[valid]
    else:
        edge_attr = None
    return edge_index, edge_attr



class VPRGraphEncoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim=256,
                 n_layers=2,
                 proj_dim=64,
                 num_node_classes=None,
                 node_emb_dim=64,
                 num_edge_classes=None,
                 edge_emb_dim=64,
                 edge_cont_dim=10,
                 dropout=0.1,
                 head=4):
        super().__init__()

        self.use_node_class = (num_node_classes is not None)
        self.use_edge_label = (num_edge_classes is not None)
        self.edge_alpha = nn.Parameter(torch.tensor(0.0))
        self.edge_cont_ln = nn.LayerNorm(hidden_dim)
        self.edge_lbl_ln = nn.LayerNorm(hidden_dim)

        self.node_emb = None
        if self.use_node_class:
            self.node_emb = nn.Embedding(num_node_classes, node_emb_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)

        self.edge_emb = None
        if self.use_edge_label:
            self.edge_emb = nn.Embedding(num_edge_classes, edge_emb_dim)
            nn.init.xavier_uniform_(self.edge_emb.weight)

        self.edge_proj = None

        self.edge_cont_mlp = nn.Sequential(
            nn.Linear(edge_cont_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        if self.use_edge_label:
            self.edge_label_proj = nn.Sequential(
                nn.Linear(edge_emb_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.edge_fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.edge_label_proj = None
            self.edge_fuse = None


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
        x = batch.x

        if self.use_node_class and hasattr(batch, 'node_class') and batch.node_class is not None:
            node_cls = batch.node_class.long().to(x.device)
            node_cls = node_cls.clamp(0, self.node_emb.num_embeddings - 1)
            node_emb = self.node_emb(node_cls)
            x = torch.cat([x, node_emb], dim=1)

        h = self.input_mlp(x)

        edge_attr = None
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            edge_attr_cont = batch.edge_attr[:,:].float().to(x.device) 
            edge_cont = self.edge_cont_mlp(edge_attr_cont)
        else:
            edge_cont = None

        if self.use_edge_label and hasattr(batch, 'edge_label') and batch.edge_label is not None:
            edge_label = batch.edge_label.long().to(x.device)
            edge_label = edge_label.clamp(0, self.edge_emb.num_embeddings - 1)
            edge_lbl = self.edge_emb(edge_label)
            edge_lbl = self.edge_label_proj(edge_lbl)

            if edge_cont is not None:
                # edge_attr = self.edge_fuse(torch.cat([edge_cont, edge_lbl], dim=1))
                # gate = self.edge_gate(torch.cat([edge_cont, edge_lbl], dim=1))
                # edge_attr = gate * edge_cont + (1 - gate) * edge_lbl
                edge_attr = edge_lbl # + self.edge_alpha * edge_cont
            else:
                edge_attr = edge_lbl

        edge_index, edge_attr = _sanitize_graph_edges(batch.edge_index, h.size(0), edge_attr)
        for conv in self.convs:
            h = conv(h, edge_index, edge_attr)
            h = self.act(h)
            h = self.drop(h)

        hg_mean = global_mean_pool(h, batch.batch)
        hg_max = global_max_pool(h, batch.batch)
        hg = torch.cat([hg_mean, hg_max], dim=1)

        z = self.proj(hg)
        z = F.normalize(z, p=2, dim=1)
        return z

    @property
    def out_dim(self):
        return self._proj_dim

class GATGraphEncoder(nn.Module):
    def __init__(self,
                 in_dim,
                 hidden_dim=256,
                 n_layers=2,
                 proj_dim=64,
                 num_node_classes=None,
                 node_emb_dim=64,
                 num_edge_classes=None,
                 edge_emb_dim=64,
                 edge_cont_dim=10,
                 dropout=0.1,
                 heads=4):
        super().__init__()

        self.use_node_class = (num_node_classes is not None)
        self.use_edge_label = (num_edge_classes is not None)
        self.edge_alpha = nn.Parameter(torch.tensor(0.0))
        self.edge_cont_ln = nn.LayerNorm(hidden_dim)
        self.edge_lbl_ln = nn.LayerNorm(hidden_dim)

        self.node_emb = None
        if self.use_node_class:
            self.node_emb = nn.Embedding(num_node_classes, node_emb_dim)
            nn.init.xavier_uniform_(self.node_emb.weight)

        self.edge_emb = None
        if self.use_edge_label:
            self.edge_emb = nn.Embedding(num_edge_classes, edge_emb_dim)
            nn.init.xavier_uniform_(self.edge_emb.weight)

        self.edge_proj = None

        self.edge_cont_mlp = nn.Sequential(
            nn.Linear(edge_cont_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.edge_gate = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid()
        )

        if self.use_edge_label:
            self.edge_label_proj = nn.Sequential(
                nn.Linear(edge_emb_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
            )

            self.edge_fuse = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            self.edge_label_proj = None
            self.edge_fuse = None


        eff_in_dim = in_dim + (node_emb_dim if self.use_node_class else 0)

        self.input_mlp = nn.Sequential(
            nn.Linear(eff_in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True)
        )

        self.convs = nn.ModuleList([
            GATv2Conv(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                heads=heads,
                concat=False,      
                dropout=dropout,
                edge_dim=hidden_dim,
                add_self_loops=True,
                residual=True,
            )
            for _ in range(n_layers)
        ])


        self.act = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(p=dropout)


        self.pool_out_dim = hidden_dim * 2
        self.proj = nn.Sequential(
            nn.Linear(self.pool_out_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, proj_dim)
        )
        self._proj_dim = proj_dim

    def forward(self, batch, return_attn=False):
        x = batch.x

        if self.use_node_class and hasattr(batch, 'node_class') and batch.node_class is not None:
            node_cls = batch.node_class.long().to(x.device)
            node_cls = node_cls.clamp(0, self.node_emb.num_embeddings - 1)
            node_emb = self.node_emb(node_cls)
            x = torch.cat([x, node_emb], dim=1)

        h = self.input_mlp(x)

        edge_attr = None
        if hasattr(batch, 'edge_attr') and batch.edge_attr is not None:
            edge_attr_cont = batch.edge_attr[:,:].float().to(x.device) 
            edge_cont = self.edge_cont_mlp(edge_attr_cont)
        else:
            edge_cont = None

        if self.use_edge_label and hasattr(batch, 'edge_label') and batch.edge_label is not None:
            edge_label = batch.edge_label.long().to(x.device)
            edge_label = edge_label.clamp(0, self.edge_emb.num_embeddings - 1)
            edge_lbl = self.edge_emb(edge_label)
            edge_lbl = self.edge_label_proj(edge_lbl)

            if edge_cont is not None:
                # edge_attr = self.edge_fuse(torch.cat([edge_cont, edge_lbl], dim=1))
                # gate = self.edge_gate(torch.cat([edge_cont, edge_lbl], dim=1))
                # edge_attr = gate * edge_cont + (1 - gate) * edge_lbl
                edge_attr = edge_lbl # + self.edge_alpha * edge_cont
            else:
                edge_attr = edge_lbl
        
        edge_index, edge_attr = _sanitize_graph_edges(batch.edge_index, h.size(0), edge_attr)
        attn_debug = None
        for i, conv in enumerate(self.convs):
            if return_attn and i == len(self.convs) - 1:
                h, attn_debug = conv(h, edge_index, edge_attr, return_attention_weights=True)
            else:
                h = conv(h, edge_index, edge_attr)

            h = self.act(h)
            h = self.drop(h)

        hg_mean = global_mean_pool(h, batch.batch)
        hg_max = global_max_pool(h, batch.batch)
        hg = torch.cat([hg_mean, hg_max], dim=1)

        z = self.proj(hg)
        z = F.normalize(z, p=2, dim=1)
        return z

    @property
    def out_dim(self):
        return self._proj_dim



class MultiModalVPRGraphEncoder(nn.Module):
    def __init__(
        self,
        graph_encoder,
        image_encoder,
        image_out_dim=8448,
        graph_out_dim=128,
        fusion_dim=256,
        normalize=True,
        graph_fusion_scale=0.05,
        freeze_image_encoder=True,
        train_only_aggregator=True,  
    ):
        super().__init__()

        self.graph_encoder = graph_encoder
        self.image_encoder = image_encoder
        self.normalize = normalize
        self.graph_fusion_scale = graph_fusion_scale
        self.freeze_image_encoder = freeze_image_encoder
        self.train_only_aggregator = train_only_aggregator

        if freeze_image_encoder and self.image_encoder is not None:
            for p in self.image_encoder.parameters():
                p.requires_grad = False
            
            
            if train_only_aggregator:
                # Разморозить только aggregator внутри image_encoder
                """
                if not hasattr(self.image_encoder, "aggregator"):
                    raise AttributeError("image_encoder has no attribute 'aggregator'")
                for p in self.image_encoder.aggregator.parameters():
                    p.requires_grad = True
                """
                for p in self.image_encoder.aggregator.linear.parameters():
                    p.requires_grad = True

        self.image_proj = nn.Sequential(
            nn.Linear(image_out_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
        )

        self.graph_proj = nn.Sequential(
            nn.Linear(graph_out_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(fusion_dim),
        )

        self.graph_gate = nn.Sequential(
            nn.Linear(graph_out_dim, fusion_dim),
            nn.Sigmoid(),
        )

        self.fuse_norm = nn.LayerNorm(fusion_dim)
        self.fuse_mlp = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
        )

        self._out_dim = fusion_dim
    def forward(self, graph=None, image=None, mode="fusion", return_parts=False):
        out = {}

        graph_z = None
        image_z = None

        if mode in ("graph", "fusion"):
            if graph is None:
                raise ValueError("graph is required for mode='graph' or 'fusion'")
            graph_z = self.graph_encoder(graph)  # [B, graph_out_dim]
            out["graph"] = graph_z

        if mode in ("image", "fusion"):
            if image is None:
                raise ValueError("image is required for mode='image' or 'fusion'")
            # if self.freeze_image_encoder:
            #    with torch.no_grad():
            #        image_raw = self.image_encoder(image)  # [B, 8448]
            #else:
            image_raw = self.image_encoder(image)

            # image_z = self.image_proj(image_raw)  # [B, fusion_dim]
            out["image"] = image_raw

        if mode == "graph":
            z = graph_z
            out["fused"] = z
            return out if return_parts else z

        if mode == "image":
            z = image_raw
            # if self.normalize:
            #    z = F.normalize(z, p=2, dim=1)
            out["fused"] = z
            return out if return_parts else z

        graph_feat = self.graph_proj(graph_z)
        gate = self.graph_gate(graph_z)

        fused = image_raw + self.graph_fusion_scale * gate * graph_z
        fused = self.fuse_norm(fused)
        fused = fused + 0.1 * self.fuse_mlp(fused)

        if self.normalize:
            fused = F.normalize(fused, p=2, dim=1)

        out["fused"] = fused
        return out if return_parts else fused

    @property
    def out_dim(self):
        return self._out_dim

class EdgeAttrNormalizer:
    def __init__(self, log_indices=None, eps=1e-6):
        self.log_indices = log_indices
        self.eps = eps

        self.count = 0
        self.mean = None
        self.M2 = None

    def _preprocess(self, x):
        x = x.clone()

        if self.log_indices:
            x[:, self.log_indices] = torch.log1p(x[:, self.log_indices])
        
        return x
    
    def update(self, x):
        if x is None or x.numel() == 0:
            return

        x = self._preprocess(x).double()

        if self.mean is None:
            self.mean = torch.zeros(x.shape[1], dtype=torch.float64)
            self.M2 = torch.zeros(x.shape[1], dtype=torch.float64)

        n = x.shape[0]

        new_count = self.count + n
        delta = x.mean(dim=0) - self.mean

        new_mean = self.mean + delta * n / new_count

        m_a = self.M2
        m_b = ((x - x.mean(dim=0))**2).sum(dim=0)

        M2 = m_a + m_b + delta**2 * self.count * n / new_count

        self.mean = new_mean
        self.M2 = M2
        self.count = new_count

    def finalize(self):
        if self.count < 2:
            raise RuntimeError("Not enough data to compute std")

        var = self.M2 / (self.count - 1)
        self.std = torch.sqrt(var).float()
        self.mean = self.mean.float()

    def transform(self, x):
        if x is None or x.numel() == 0:
            return x

        x = self._preprocess(x)
        return (x - self.mean) / (self.std + self.eps)