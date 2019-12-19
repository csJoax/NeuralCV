""" Parts of the U-Net model """

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ["DoubleConv", "Down", "Up", "Concat", "OutConv"]


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.MaxPool2d):
    """Downscaling with maxpool"""

    def __init__(self):
        super(Down, self).__init__(2)


class Up(nn.Module):
    """Upscaling()"""

    def __init__(self, in_channels=None, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(
                scale_factor=2, mode='bilinear', align_corners=True)
        else:
            if not in_channels:
                raise ValueError("in_channels required.")
            
            self.up = nn.ConvTranspose2d(
                in_channels // 2, in_channels // 2, kernel_size=2, stride=2)

    def forward(self, x):
        x = self.up(x)
        return x


def Concat(nn.Module):
    """align tensors then torch.cat"""

    def forward(self, *inputs):
        # input is CHW
        sizes = [x.size() for x in inputs]

        height = max(size[2] for size in sizes)
        width = max(size[3] for size in sizes)

        padded_inputs = []
        for input, size in zip(inputs, sizes):
            diffY = height - size[2]
            diffX = width - size[3]
            if diffY or diffX:
                padXL, padXR = diffX // 2, diffX - diffX // 2
                padYL, padYR = diffY // 2, diffY - diffY // 2

                x = F.pad(input, [padXL, padXR,
                                  padYL, padYR])
            else:
                x = input

            padded_inputs.append(x)

        x = torch.cat(padded_inputs, dim=1)
        return x


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
