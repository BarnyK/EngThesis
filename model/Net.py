import torch
from torch.nn import functional as F
from torch import nn
from .blocks import (
    StackedHourglassModule,
    FeatureExtraction,
)
from .utils import init_convolutions

class Net(nn.Module):
    ## Main module for Network
    def __init__(self, max_disp=192, no_sdea=False):
        super().__init__()
        self.maxdisp = max_disp
        self.feature_extraction = FeatureExtraction(max_disp, no_sdea)

        self.stacked_hourglass = StackedHourglassModule()

        self.disparities = torch.arange(max_disp, requires_grad=False).reshape(
            (1, max_disp, 1, 1)
        )

        for m in self.modules():
            if isinstance(m,(nn.Conv2d,nn.Conv3d)):
                init_convolutions(m)

    def to(self, device):
        """
            Additonaly move the self.disparities array to given device.
        """
        new_self = super().to(device)
        new_self.disparities = new_self.disparities.to(device)

        return new_self

    def forward(self, left, right):
        left, right = self.feature_extraction(left, right)

        # 32 x H/4 x W/4
        cost = self.create_cost_volume(left, right)

        # 64 x maxdisp / 4 x H/4 x W/4
        out1, out2, out3 = self.stacked_hourglass(cost)

        # 1 x maxdisp / 4 x H/4 x W/4
        out3 = self.upsample_regression(out3)
        
        # H x W
        if not self.training:
            return out3

        out1 = self.upsample_regression(out1)
        out2 = self.upsample_regression(out2)
        return out1, out2, out3

    def upsample_regression(self, input):
        # Upsample
        out = F.interpolate(input, scale_factor=4, mode="trilinear", align_corners=True)
        # Remove the channels dimension, whch is 1 at this point
        out = torch.squeeze(out, 1)
        # Softmax over dimension with disparity
        out = F.softmax(out, 1)
        # Multiply weight by range of disparities used
        out = out * self.disparities
        # Sum to get the final disparity
        out = torch.sum(out, 1)
        return out

    def create_cost_volume(self, left: torch.Tensor, right: torch.Tensor):
        # Initialize volume with zeros on the same device as input
        cost = torch.zeros(
            (
                left.shape[0],
                left.shape[1] + right.shape[1],
                self.maxdisp // 4,
                left.shape[2],
                left.shape[3],
            ),
            device=left.device,
            dtype=left.dtype,
        )
        
        # Copy from feature matrices to cost volume
        ch = left.shape[1]
        cost[:, :ch, 0, :, :] = left
        cost[:, ch:, 0, :, :] = right
        for i in range(1, self.maxdisp // 4):
            cost[:, :ch, i, :, i:] = left[:, :, :, i:]
            cost[:, ch:, i, :, i:] = right[:, :, :, :-i]
        return cost
