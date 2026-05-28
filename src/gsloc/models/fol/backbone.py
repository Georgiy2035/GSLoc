"""DINOv2 backbone (vendored from https://github.com/chenshunpeng/FoL)."""

from __future__ import annotations

import torch
import torch.nn as nn

DINOV2_ARCHS = {
    "dinov2_vits14": 384,
    "dinov2_vitb14": 768,
    "dinov2_vitl14": 1024,
    "dinov2_vitg14": 1536,
}


class DINOv2(nn.Module):
    def __init__(
        self,
        model_name: str = "dinov2_vitb14",
        num_trainable_blocks: int = 2,
        norm_layer: bool = False,
        return_token: bool = False,
        pretrained: bool = False,
    ) -> None:
        super().__init__()
        if model_name not in DINOV2_ARCHS:
            raise ValueError(f"Unknown model name {model_name}")
        # FoL checkpoint already contains finetuned backbone weights; skip ~330MB DINOv2 download.
        self.model = torch.hub.load("facebookresearch/dinov2", model_name, pretrained=pretrained)
        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x: torch.Tensor):
        b, _c, h, w = x.shape
        x = self.model.prepare_tokens_with_masks(x)

        with torch.no_grad():
            for blk in self.model.blocks[: -self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        for i, blk in enumerate(self.model.blocks[-self.num_trainable_blocks :]):
            if i == 3:
                y = blk.norm1(x)
                b_n, n, c = y.shape
                qkv = (
                    blk.attn.qkv(y)
                    .reshape(b_n, n, 3, blk.attn.num_heads, c // blk.attn.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, _v = qkv[0], qkv[1], qkv[2]
                att = (q @ k.transpose(-2, -1)) * blk.attn.scale
                att = att.softmax(dim=-1)
                att_cls_to_patches = att[:, :, 0, 1:]
                att_cls_to_patches_mean = att_cls_to_patches.sum(dim=1)
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]
        f = x[:, 1:]
        f = f.reshape((b, h // 14, w // 14, self.num_channels)).permute(0, 3, 1, 2)
        if self.return_token:
            return f, t, att_cls_to_patches_mean
        return f
