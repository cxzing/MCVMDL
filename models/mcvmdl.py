import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
Original version.
"""

class SourceEncoder(nn.Module):
    def __init__(self, in_channels=1, cond_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv3d(in_channels, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(2),
            nn.Conv3d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d(1)
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16, cond_dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, source):
        return self.fc(self.net(source))   #



class FiLM3D(nn.Module):
    def __init__(self, num_features, cond_dim):
        super().__init__()
        self.gamma = nn.Linear(cond_dim, num_features)
        self.beta = nn.Linear(cond_dim, num_features)

    def forward(self, feat, cond):
        # feat: [B, C, D, H, W]
        gamma = self.gamma(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        beta = self.beta(cond).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        return feat * (1 + gamma) + beta
class DoubleConv3D(nn.Module):
    """
    3D convolution -> ReLU -> 3D convolution -> ReLU.
    """

    def __init__(self, in_channels, out_channels):
        super(DoubleConv3D, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet3D_SourceMod(nn.Module):
    def __init__(self, in_channels, out_channels, cond_dim=128, base_channels=32):
        super().__init__()

        if base_channels <= 0:
            raise ValueError("base_channels must be a positive integer")

        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8
        cb = c1 * 16

        tissue_channels = in_channels - 1
        self.source_encoder = SourceEncoder(in_channels=1, cond_dim=cond_dim)

        self.down1 = DoubleConv3D(in_channels, c1)
        self.down2 = DoubleConv3D(c1, c2)
        self.down3 = DoubleConv3D(c2, c3)
        self.down4 = DoubleConv3D(c3, c4)
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = DoubleConv3D(c4, cb)


        self.filmb = FiLM3D(cb, cond_dim)

        self.up4 = nn.ConvTranspose3d(cb, c4, 2, 2)
        self.conv4 = DoubleConv3D(c4 * 2, c4)
        self.up3 = nn.ConvTranspose3d(c4, c3, 2, 2)
        self.conv3 = DoubleConv3D(c3 * 2, c3)
        self.up2 = nn.ConvTranspose3d(c3, c2, 2, 2)
        self.conv2 = DoubleConv3D(c2 * 2, c2)
        self.up1 = nn.ConvTranspose3d(c2, c1, 2, 2)
        self.conv1 = DoubleConv3D(c1 * 2, c1)



        self.final = nn.Conv3d(c1, out_channels, 1)

    def forward(self, x):
        source = x[:, -1:, ...]
        cond = self.source_encoder(source)



        x1 = self.down1(x)
        x2 = self.down2(self.pool(x1))
        x3 = self.down3(self.pool(x2))
        x4 = self.down4(self.pool(x3))

        b = self.bottleneck(self.pool(x4))
        b = self.filmb(b, cond)

        u4 = self.up4(b)
        u4 = torch.cat([u4, x4], dim=1)
        u4 = self.conv4(u4)

        u3 = self.up3(u4)
        u3 = torch.cat([u3, x3], dim=1)
        u3 = self.conv3(u3)

        u2 = self.up2(u3)
        u2 = torch.cat([u2, x2], dim=1)
        u2 = self.conv2(u2)

        u1 = self.up1(u2)
        u1 = torch.cat([u1, x1], dim=1)
        u1 = self.conv1(u1)
        out = torch.sigmoid(self.final(u1))



        return out






