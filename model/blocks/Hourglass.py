import torch
import torch.nn.functional as F
from torch import nn
from ..utils import conv3d_norm, conv3d_norm_relu


class Hourglass(nn.Module):
    def __init__(self, in_layers):
        super().__init__()
        self.conv1 = conv3d_norm_relu(in_layers, in_layers * 2, 3, 2, 1)

        # Conv2 doesn't have relu because of the skip connection
        self.conv2 = conv3d_norm(in_layers * 2, in_layers * 2, 3, 1, 1)

        self.conv3 = conv3d_norm_relu(in_layers * 2, in_layers * 2, 3, 2, 1)

        self.conv4 = conv3d_norm_relu(in_layers * 2, in_layers * 2, 3, 1, 1)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose3d(in_layers * 2, in_layers * 2, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(in_layers * 2),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose3d(in_layers * 2, in_layers, 3, 2, 1, 1, bias=False),
            nn.BatchNorm3d(in_layers),
        )

    def forward(
        self,
        input: torch.Tensor,
        skip1: torch.Tensor = None,
        skip2: torch.Tensor = None,
    ):
        out = self.conv1(input)

        out = self.conv2(out)
        if skip1 is not None:
            out = out + skip1
        out = F.relu(out)
        out_skip1 = out

        out = self.conv3(out)

        out = self.conv4(out)

        out = self.deconv1(out)
        if skip2 is not None:
            out = out + skip2
        else:
            out = out + out_skip1  # happens on first hourglass
        out = F.relu(out)
        out_skip2 = out

        out = self.deconv2(out)

        return out, out_skip1, out_skip2
