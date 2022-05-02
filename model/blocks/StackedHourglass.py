from torch import nn

from ..utils import conv3d_norm, conv3d_norm_relu
from .Hourglass import Hourglass


class StackedHourglass(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            conv3d_norm_relu(64, 32, 3, 1, 1),
            conv3d_norm_relu(32, 32, 3, 1, 1),
        )

        self.conv2 = nn.Sequential(
            conv3d_norm_relu(32, 32, 3, 1, 1),
            conv3d_norm(32, 32, 3, 1, 1),
        )
        self.hourglass1 = Hourglass(32)
        self.hourglass2 = Hourglass(32)
        self.hourglass3 = Hourglass(32)

        self.hourglass1_processing = nn.Sequential(
            conv3d_norm_relu(32, 32, 3, 1, 1), nn.Conv3d(32, 1, 3, 1, 1, bias=False)
        )
        self.hourglass2_processing = nn.Sequential(
            conv3d_norm_relu(32, 32, 3, 1, 1), nn.Conv3d(32, 1, 3, 1, 1, bias=False)
        )
        self.hourglass3_processing = nn.Sequential(
            conv3d_norm_relu(32, 32, 3, 1, 1), nn.Conv3d(32, 1, 3, 1, 1, bias=False)
        )

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out) + out

        skip_connection = out

        out1, skip1, skip2 = self.hourglass1(out)
        out1 += skip_connection

        out2, _, skip2 = self.hourglass2(out1, skip2, skip1)
        out2 += skip_connection

        out3, *_ = self.hourglass3(out2, skip2, skip1)
        out3 += skip_connection

        out1 = self.hourglass1_processing(out1)
        out2 = self.hourglass2_processing(out2) + out1
        out3 = self.hourglass3_processing(out3) + out2

        return out1, out2, out3
