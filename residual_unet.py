import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm1d(out_channels)
        self.residual_block = ResidualBlock(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.residual_block(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UpBlock, self).__init__()
        self.upconv = nn.ConvTranspose1d(in_channels, out_channels, kernel_size=2, stride=2)
        self.residual_block = ResidualBlock(out_channels + skip_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        x = torch.cat((x, skip), dim=1)  # Concatenate along the channel dimension
        x = self.residual_block(x)
        return x

class ResidualUNet(nn.Module):
    def __init__(self, in_channels=128, out_channels=1):
        super(ResidualUNet, self).__init__()
        self.down1 = DownBlock(in_channels, 16)
        self.down2 = DownBlock(16, 32)
        self.down3 = DownBlock(32, 64)
        self.bottleneck = ResidualBlock(64)
        self.up3 = UpBlock(64, 32, 32)
        self.up2 = UpBlock(32 + 32, 16, 16)
        self.up1 = UpBlock(16 + 16, out_channels, in_channels)

    def forward(self, x):
        skip1 = self.down1(x)
        skip2 = self.down2(skip1)
        skip3 = self.down3(skip2)
        x = self.bottleneck(skip3)
        x = self.up3(x, skip3)
        x = self.up2(x, skip2)
        x = self.up1(x, skip1)
        return x
