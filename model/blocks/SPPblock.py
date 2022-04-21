from tokenize import group
import torch
import torch.nn.functional as F
from torch import nn

from ..utils import conv2d_norm_relu


class SPPBlock(nn.Module):
    def __init__(self, input_channels, pool_output_channels, skip_channels):
        super().__init__()

        self.pool1 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(64, 64), stride=64),
            conv2d_norm_relu(input_channels, pool_output_channels, 1, 1,groups=2),
        )
        self.pool2 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(32, 32), stride=32),
            conv2d_norm_relu(input_channels, pool_output_channels, 1, 1,groups=2),
        )
        self.pool3 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(16, 16), stride=16),
            conv2d_norm_relu(input_channels, pool_output_channels, 1, 1,groups=2),
        )
        self.pool4 = nn.Sequential(
            nn.AvgPool2d(kernel_size=(8, 8), stride=8),
            conv2d_norm_relu(input_channels, pool_output_channels, 1, 1,groups=2),
        )

        ## concatenation of all 4 pools + input + skip connection happens here 128, 32, 64
        concat_channels = pool_output_channels * 4 + input_channels + skip_channels
        self.conv = nn.Sequential(
            conv2d_norm_relu(concat_channels, 128, 3, 1, 1,groups=8),
            nn.Conv2d(128, 32, 1, 1,bias=False),
        )

    def forward(self, input, skip_connection):
        h, w = skip_connection.shape[2:]

        pool1_result = F.interpolate(
            self.pool1(input), (h, w), mode="bilinear", align_corners=True
        )
        pool2_result = F.interpolate(
            self.pool2(input), (h, w), mode="bilinear", align_corners=True
        )
        pool3_result = F.interpolate(
            self.pool3(input), (h, w), mode="bilinear", align_corners=True
        )
        pool4_result = F.interpolate(
            self.pool4(input), (h, w), mode="bilinear", align_corners=True
        )

        concat = torch.cat(
            (
                skip_connection, # 64
                input,           # 128
                pool4_result,    # 32
                pool3_result,    # 32
                pool2_result,    # 32 
                pool1_result,    # 32
            ),
            dim=1,
        )
        return self.conv(concat)
