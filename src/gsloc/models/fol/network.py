"""FoL network (vendored from https://github.com/chenshunpeng/FoL)."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from gsloc.models.fol.aggregator import FoLAggregator
from gsloc.models.fol.backbone import DINOv2


class FoLNet(nn.Module):
    """Backbone + FoL aggregation (ViT-B base variant by default)."""

    def __init__(
        self,
        num_channels: int = 768,
        model_name: str = "dinov2_vitb14",
        num_trainable_blocks: int = 4,
        backbone_pretrained: bool = False,
    ) -> None:
        super().__init__()
        self.backbone = DINOv2(
            model_name=model_name,
            num_trainable_blocks=num_trainable_blocks,
            return_token=True,
            norm_layer=True,
            pretrained=backbone_pretrained,
        )
        self.aggregator = FoLAggregator(
            num_channels=num_channels,
            num_clusters=64,
            cluster_dim=128,
            token_dim=256,
        )
        self.upconv = nn.ConvTranspose2d(in_channels=num_channels, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.upconv2 = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, test: bool = False):
        x = self.backbone(x)
        mask1 = x[2]
        mask_guide = torch.where(mask1 >= 0.01, mask1, torch.zeros_like(mask1))
        x, local_f, mask, loss_kl_1, loss_kl_2, mix_matrix, weak_supervision_info = self.aggregator(
            x, mask_guide, test=test
        )

        x0 = self.upconv(local_f)
        x0 = self.relu(x0)
        x0 = self.upconv2(x0)
        x0 = x0.permute(0, 2, 3, 1)
        local_all = F.normalize(x0, p=2, dim=-1)
        local_feature_separate = F.normalize(x0.detach(), p=2, dim=-1)

        bs, h, w, _c = x0.shape
        mask2 = mix_matrix.reshape(bs, mask.shape[1], mask.shape[2]).float()
        mask_interpolated = (
            F.interpolate(mask2.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=True).squeeze(1)
        )
        mask_separate = mask_interpolated
        mask = F.interpolate(mask.unsqueeze(1), size=(h, w), mode="nearest").squeeze(1)
        local_f_flat = x0.view(x0.size(0), -1, x0.size(-1))
        mask_flat = mask.view(mask.size(0), -1)
        indices = mask_flat.nonzero(as_tuple=True)
        selected_batch_tokens = local_f_flat[indices]
        split_sizes = mask_flat.sum(dim=1).to(torch.int).tolist()
        selected_tokens = selected_batch_tokens.split(split_sizes)

        if h > 80:
            max_len = 3600
        elif h > 70:
            max_len = 3500
        elif h > 40:
            max_len = 1000
        else:
            max_len = h * h

        selected_tokens = list(selected_tokens)
        padded_first_token = torch.cat(
            [
                selected_tokens[0],
                torch.zeros(
                    (max_len - selected_tokens[0].size(0), selected_tokens[0].size(1)),
                    device=selected_tokens[0].device,
                ),
            ],
            dim=0,
        )
        selected_tokens[0] = padded_first_token
        padded_tokens = torch.nn.utils.rnn.pad_sequence(selected_tokens, batch_first=True, padding_value=0)
        local_feature = F.normalize(padded_tokens, p=2, dim=-1)
        if weak_supervision_info is not None:
            weak_supervision_info.append(local_all)
        return (
            x,
            local_feature,
            loss_kl_1,
            loss_kl_2,
            [local_feature_separate, mask_separate],
            weak_supervision_info,
            mask_guide,
        )
