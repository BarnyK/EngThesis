from torch import nn


class BaseBlock(nn.Module):
    def __init__(
        self, in_channels, out_channels, kernel=3, stride=1, padding=1, dilation=1
    ):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels, out_channels, kernel, stride, padding, dilation, bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel, 1, padding, bias=False),
            nn.BatchNorm2d(out_channels),
        )

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride),
                nn.BatchNorm2d(out_channels),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        out = self.block(input)

        skip = input
        if self.downsample != None:
            skip = self.downsample(input)

        out += skip
        out = self.relu(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, blocks, in_channels, out_channels, stride):
        super().__init__()
        layers = []
        cur_channels = in_channels
        cur_stride = stride

        for i in range(blocks):
            layer = BaseBlock(
                cur_channels,
                out_channels,
                3,
                cur_stride,
            )
            layers.append(layer)
            cur_stride = 1
            cur_channels = out_channels

        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)
