import torch.nn as nn

from mmpr.modules import NetVLAD, REM


class REIN(nn.Module):
    def __init__(self):
        super(REIN, self).__init__()
        self.rem = REM()
        self.pooling = NetVLAD()

        self.local_feat_dim = 128
        self.global_feat_dim = self.local_feat_dim * 64

    def forward(self, x):
        out1, local_feats = self.rem(x)

        global_desc = self.pooling(out1)

        return out1, local_feats, global_desc
