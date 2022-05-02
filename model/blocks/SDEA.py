from time import time
import torch
import torch.nn.functional as F
from torch import nn


class SDEABlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, max_disp: int):
        super().__init__()
        self.max_disp = max_disp
        self.g1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.GroupNorm(16, out_channels),
            nn.ReLU(),
        )
        self.g2 = nn.Conv2d(out_channels, 1, 1, 1, 0)

        self.channel_scaling = None
        if in_channels != out_channels:
            self.channel_scaling = nn.Conv2d(in_channels, out_channels, 1, 1)

    def forward(self, left, right):
        # two convolution layers G1
        left_g1 = self.g1(left)
        right_g1 = self.g1(right)

        # scaling the input channels
        if self.channel_scaling:
            left = self.channel_scaling(left)
            right = self.channel_scaling(right)

        # 1x1 convolution layer G2
        left_g2 = self.g2(left_g1)
        right_g2 = self.g2(right_g1)

        # Weight calculation
        left_weight = wmc(left_g2, right_g2, self.max_disp)
        right_weight = wmc(right_g2, left_g2, self.max_disp)

        # Elementwise multiplication
        left_out = left_g1 * left_weight
        right_out = right_g1 * right_weight

        # Elementwise sum
        left_out = left_out + left
        right_out = right_out + right

        return left_out, right_out


@torch.jit.script
def weight_matrix_calculation(left, right, max_disp: int):
    weight_volume = torch.empty(
        (
            left.shape[0],
            max_disp * 2 + 1,
            left.shape[2],
            left.shape[3],
        ),
        device=left.device,
        dtype=left.dtype,
    )

    weight_volume[:, :, :, :] = left - right
    for i in range(1, max_disp + 1):
        weight_volume[:, 2 * i - 1, :, :-i] = left[:, 0, :, :-i] - right[:, 0, :, i:]
        weight_volume[:, 2 * i, :, i:] = left[:, 0, :, i:] - right[:, 0, :, :-i]

    weight_volume = weight_volume.abs()
    weight_volume, _ = weight_volume.min(1, keepdim=True)
    weight_volume = 1 - torch.sigmoid(weight_volume)
    return weight_volume
