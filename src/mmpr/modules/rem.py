import torch
import torch.nn as nn
import torch.nn.functional as F

import torchvision.models as models


class REM(nn.Module):
    def __init__(self, from_scratch=False, rotations=8):
        super(REM, self).__init__()

        # cnn backbone
        pretrain = not from_scratch
        encoder = models.resnet34(pretrained=pretrain)  # resnet34
        layers = list(encoder.children())[:-4]
        self.encoder = nn.Sequential(*layers)

        # rotations
        self.angles = -torch.arange(0, 359.00001, 360.0 / rotations) / 180 * torch.pi

    def forward(self, x):
        equ_features = []

        batch_size = x.size(0)

        for i in range(len(self.angles)):
            # input warp grids
            aff = torch.zeros(batch_size, 2, 3).cuda()
            aff[:, 0, 0] = torch.cos(-self.angles[i])
            aff[:, 0, 1] = torch.sin(-self.angles[i])
            aff[:, 1, 0] = -torch.sin(-self.angles[i])
            aff[:, 1, 1] = torch.cos(-self.angles[i])
            grid = F.affine_grid(aff, torch.Size(x.size()), align_corners=True).type(x.type())

            # input warp
            warped_im = F.grid_sample(x, grid, align_corners=True, mode="bicubic")

            # cnn backbone feature
            out = self.encoder(warped_im)

            # output feature warp grids
            if i == 0:
                im1_init_size = out.size()

            aff = torch.zeros(batch_size, 2, 3).cuda()
            aff[:, 0, 0] = torch.cos(self.angles[i])
            aff[:, 0, 1] = torch.sin(self.angles[i])
            aff[:, 1, 0] = -torch.sin(self.angles[i])
            aff[:, 1, 1] = torch.cos(self.angles[i])
            grid = F.affine_grid(aff, torch.Size(im1_init_size), align_corners=True).type(x.type())

            # output feature warp
            out = F.grid_sample(out, grid, align_corners=True, mode="bicubic")

            equ_features.append(out.unsqueeze(-1))

        equ_features = torch.cat(equ_features, axis=-1)  # B C H W R

        B, C, H, W, R = equ_features.shape
        equ_features = torch.max(equ_features, dim=-1, keepdim=False)[0]  # max pooling along rotations

        aff = torch.zeros(batch_size, 2, 3).cuda()
        aff[:, 0, 0] = 1
        aff[:, 0, 1] = 0
        aff[:, 1, 0] = 0
        aff[:, 1, 1] = 1

        # upsample for NetVLAD
        B, C, H, W = x.size()
        grid = F.affine_grid(aff, torch.Size((B, C, H // 4, W // 4)), align_corners=True).type(
            x.type()
        )  # ,align_corners=True)
        out1 = F.grid_sample(equ_features, grid, align_corners=True, mode="bicubic")
        out1 = F.normalize(out1, dim=1)

        # upsample for keypoints
        grid = F.affine_grid(aff, torch.Size((B, C, H, W)), align_corners=True).type(
            x.type()
        )  # ,align_corners=True)
        out2 = F.grid_sample(equ_features, grid, align_corners=True, mode="bicubic")
        out2 = F.normalize(out2, dim=1)

        return out1, out2
