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
        # Upsampling blocks
        self.up3 = UpBlock(64, 32, 32)  # Upsampled channels + skip connection channels
        self.up2 = UpBlock(32+32, 16, 16)  # Channels from previous UpBlock and skip connection
        self.up1 = UpBlock(16+16, out_channels, in_channels)  # Final upsample to match input channels

    def forward(self, x):
        # Downsampling
        skip1 = self.down1(x)  # Output here has 16 channels
        skip2 = self.down2(skip1)  # Output here has 32 channels
        skip3 = self.down3(skip2)  # Output here has 64 channels
        # Bottleneck
        x = self.bottleneck(skip3)  # Output here has 64 channels
        # Upsampling with skip connections
        x = self.up3(x, skip2)  # Skip connection adds 32 channels
        x = self.up2(x, skip1)  # Skip connection adds 16 channels
        x = F.interpolate(x, size=skip1.size()[2:], mode='linear', align_corners=False)  # Resize to match skip1
        x = self.up1(x, skip1)  # Skip connection adds in_channels channels from the original input
        return x
