"""FoL aggregator (vendored from https://github.com/chenshunpeng/FoL)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def log_sinkhorn_iterations(
    z: torch.Tensor, log_mu: torch.Tensor, log_nu: torch.Tensor, iters: int
) -> torch.Tensor:
    u, v = torch.zeros_like(log_mu), torch.zeros_like(log_nu)
    for _ in range(iters):
        u = log_mu - torch.logsumexp(z + v.unsqueeze(1), dim=2)
        v = log_nu - torch.logsumexp(z + u.unsqueeze(2), dim=1)
    return z + u.unsqueeze(2) + v.unsqueeze(1)


def log_optimal_transport(scores: torch.Tensor, alpha: torch.Tensor, iters: int) -> torch.Tensor:
    b, m, n = scores.shape
    one = scores.new_tensor(1)
    ms, ns, bs = (m * one).to(scores), (n * one).to(scores), ((n - m) * one).to(scores)
    bins = alpha.unsqueeze(1)
    couplings = torch.cat([scores, bins], 1)
    norm = -(ms + ns).log()
    log_mu = torch.cat([norm.expand(m), bs.log()[None] + norm])
    log_nu = norm.expand(n)
    log_mu, log_nu = log_mu[None].expand(b, -1), log_nu[None].expand(b, -1)
    z = log_sinkhorn_iterations(couplings, log_mu, log_nu, iters)
    return z - norm


class FoLAggregator(nn.Module):
    def __init__(
        self,
        num_channels: int = 768,
        num_clusters: int = 64,
        cluster_dim: int = 128,
        token_dim: int = 256,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_channels = num_channels
        self.num_clusters = num_clusters
        self.cluster_dim = cluster_dim
        self.token_dim = token_dim
        dropout_layer = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.token_features = nn.Sequential(
            nn.Linear(self.num_channels, 512),
            nn.ReLU(),
            nn.Linear(512, self.token_dim),
        )
        self.cluster_features = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.cluster_dim, 1),
        )
        self.score = nn.Sequential(
            nn.Conv2d(self.num_channels, 512, 1),
            dropout_layer,
            nn.ReLU(),
            nn.Conv2d(512, self.num_clusters, 1),
        )

    @staticmethod
    def replace_top_values(important_matrix: torch.Tensor) -> torch.Tensor:
        num_change = 3
        top_values, top_indices = torch.topk(important_matrix, num_change, dim=1, largest=True, sorted=False)
        sorted_values, _sorted_indices = important_matrix.sort(dim=1, descending=True)
        k_value_index = int(0.05 * important_matrix.size(1))
        kth_values = sorted_values[:, k_value_index].unsqueeze(1)
        kth_values_expanded = kth_values.expand(-1, num_change)
        important_matrix.scatter_(1, top_indices, kth_values_expanded)
        return important_matrix

    def forward(self, x, mask, test: bool = False):
        x, t, important_matrix = x[0], x[1], x[2]
        bs, _c, h, w = x.shape
        local_f = x
        f = self.cluster_features(x).flatten(2)
        p = self.score(x).flatten(2)
        t = self.token_features(t)

        p = log_optimal_transport(p, 1 - mask, 3)
        p = torch.exp(p)
        p = p[:, :-1, :]
        if test:
            weak_supervision_info = None
        else:
            weak_supervision_info = [
                F.normalize(x.reshape(bs, -1, 529), p=2, dim=1),
                p,
            ]

        confidence_matrix = torch.mean(p, dim=1)
        important_matrix = self.replace_top_values(important_matrix)
        confidence_matrix = self.replace_top_values(confidence_matrix)
        confidence_matrix = confidence_matrix.softmax(dim=-1)
        important_matrix = important_matrix.softmax(dim=-1)

        loss_kl_1 = nn.KLDivLoss(reduction="none")(confidence_matrix.log(), important_matrix.detach()).sum(
            -1
        ).mean()
        loss_kl_2 = nn.KLDivLoss(reduction="none")(important_matrix.log(), confidence_matrix.detach()).sum(
            -1
        ).mean()

        mix_matrix = confidence_matrix if test else confidence_matrix + important_matrix

        num_select = 225
        _, topk_indices = torch.topk(mix_matrix, num_select, dim=-1)
        mask_out = torch.zeros((bs, h * w), dtype=torch.float32, device=x.device)
        mask_out.scatter_(1, topk_indices, 1)
        mask_out = mask_out.view(bs, h, w)

        p = p.unsqueeze(1).repeat(1, self.cluster_dim, 1, 1)
        f = f.unsqueeze(2).repeat(1, 1, self.num_clusters, 1)
        f = torch.cat(
            [
                F.normalize(t, p=2, dim=-1),
                F.normalize((f * p).sum(dim=-1), p=2, dim=1).flatten(1),
            ],
            dim=-1,
        )
        return (
            F.normalize(f, p=2, dim=-1),
            local_f,
            mask_out,
            loss_kl_1,
            loss_kl_2,
            mix_matrix,
            weak_supervision_info,
        )
