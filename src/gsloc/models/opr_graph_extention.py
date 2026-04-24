import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data, Batch, HeteroData
from torch_geometric.nn import GCNConv, GINEConv, global_mean_pool, global_max_pool

from gsloc.models.graph_encoder import VPRGraphEncoder
from gsloc.models.graph_encoder import MultiModalVPRGraphEncoder

class OPR_VPRGraphEncoder(VPRGraphEncoder):
    def __init__(self,
                 in_dim,
                 hidden_dim=256,
                 n_layers=2,
                 proj_dim=64,
                 num_node_classes=None,
                 node_emb_dim=64,
                 num_edge_classes=None,
                 edge_emb_dim=64,
                 dropout=0.1):

        super().__init__(
            in_dim, 
            hidden_dim, 
            n_layers, 
            proj_dim, 
            num_node_classes, 
            node_emb_dim, 
            num_edge_classes, 
            edge_emb_dim, 
            dropout)


    def forward(self, batch):
        if isinstance(batch, dict):
            batch = batch["graphs_main"]

        z = super().forward(batch)

        output = {"final_descriptor": z}
        return output


    @property
    def out_dim(self):
        return self._proj_dim



class OPR_MultiModalVPRGraphEncoder(MultiModalVPRGraphEncoder):
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
        mode="fusion"
    ):
        super().__init__(
            graph_encoder,
            image_encoder,
            image_out_dim,
            graph_out_dim,
            fusion_dim,
            normalize,
            graph_fusion_scale,
            freeze_image_encoder,
            train_only_aggregator
        )
        self.mode = mode


    def forward(self, batch, return_parts=False):
        out = {}

        graph = batch.get("graphs_main", None)
        image = batch.get("images_main", None)

        out = super().forward(graph=graph, image=image, mode=self.mode, return_parts=return_parts)
        
        result = {}
        if return_parts:
            result["graph"] = out["graph"]
            result["image"] = out["image"]
        result["final_descriptor"] = out["fused"] if return_parts else out
        return out

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
        if self.mean is None or self.std is None:
            return x
        mean = torch.as_tensor(self.mean, dtype=x.dtype, device=x.device)
        std = torch.as_tensor(self.std, dtype=x.dtype, device=x.device)
        if mean.shape[0] != x.shape[1]:
            raise ValueError(
                f"EdgeAttrNormalizer: mean/std dim {mean.shape[0]} != edge_attr dim {x.shape[1]}"
            )
        return (x - mean) / (std + self.eps)