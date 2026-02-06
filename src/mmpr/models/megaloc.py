from __future__ import annotations

from torch import nn, Tensor
import torch


class MegaLoc(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.hub.load("gmberton/MegaLoc", "get_trained_model")

    def forward(self, batch: dict[str, Tensor]) -> dict[str, Tensor]:
        key = next((k for k in batch if k.startswith("images_")), None)
        if key is None:
            raise KeyError("No key starting with 'images_' found in the batch.")
        images = batch[key]
        descriptor = self.model(images)
        output = {"final_descriptor": descriptor}
        return output
