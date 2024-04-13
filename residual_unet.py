import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=1, padding=3)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, internal_channels=None):
        super(ResidualBlock, self).__init__()
        if internal_channels is None:
            internal_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels, internal_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(internal_channels)
        self.conv2 = nn.Conv2d(internal_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return F.relu(out)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.residual_block = ResidualBlock(out_channels)

    def forward(self, x):
        x = F.relu(self.bn(self.conv(x)))
        x = self.residual_block(x)
        return x

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels):
        super(UpBlock, self).__init__()
        # Added output_padding=1 to correct the off-by-one error
        self.upconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.residual_block = ResidualBlock(out_channels + skip_channels, out_channels)

    def forward(self, x, skip):
        x = self.upconv(x)
        print("Upconv x = ", x.shape)
        # The output_padding should correct the shape to [32, 64, 128, 128]
        x = torch.cat((x, skip), dim=1)  # Concatenate along the channel dimension
        print("After concat x = ", x.shape)
        x = self.residual_block(x)
        return x

class ResidualUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=3):
        super(ResidualUNet, self).__init__()
        # Assume the number of filters just before flattening is 256
        # and the spatial dimension of the feature map at that point is 64x64

        # Calculate the number of flattened features
        flattened_size = 256 * 64 * 64  # Adjust based on your actual feature map size before flattening
        
        # Initialize network blocks with the correct sizes
        self.conv_block1 = ConvBlock(in_channels, 64)
        self.down1 = DownBlock(64, 128)
        self.down2 = DownBlock(128, 256)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(flattened_size, 256)  
        self.res_blocks = nn.Sequential(*[ResidualBlock(256) for _ in range(9)])
        self.up1 = UpBlock(256, 128, 128)
        self.up2 = UpBlock(128 + 128, 64, 64)
        self.out_conv = nn.Conv2d(64 + 64, out_channels, kernel_size=7, padding=3)

    def forward(self, x):
        # Downsample
        print("x shape = ", x.shape)
        x1 = self.conv_block1(x)
        print("x1 shape = ", x1.shape)
        x2 = self.down1(x1)
        print("x2 shape = ", x2.shape)
        x3 = self.down2(x2)
        print("x3 shape = ", x3.shape)
        
        # Flatten and pass through a dense layer
        # x_flat = self.flat(x3)
        # print("x_flat shape = ", x_flat.shape)

        # x_dense = self.dense(x_flat)
        # print("x_dense shape = ", x_dense.shape)

        # # Reshape back to feature map and apply residual blocks
        # x_res = x_dense.view(32, 128, 128, 128)  # Now the reshape should work
        x_res = self.res_blocks(x3)
        print("x_res shape = ", x_res.shape)

        # Upsample with skip connections
        x = self.up1(x_res, x2)
        print("x_up2 shape = ", x.shape)
        x = self.up2(x, x1)
        print("x_up1 shape = ", x.shape)

        # Output convolution
        x = self.out_conv(x)
        print("x shape = ", x.shape)
        return x
