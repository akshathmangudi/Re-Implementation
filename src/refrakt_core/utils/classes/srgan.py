import math
import torch
from torch import nn
from refrakt_core.utils.classes.resnet import ResidualBlock


class UpsampleBlock(nn.Module):
    """
    Optimized upsampling block using ConvTranspose2d.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.upsample(x)


class SRResidualBlock(ResidualBlock):
    """
    A modified version of ResidualBlock specifically for Super Resolution.
    Inherits from the base ResidualBlock but adapts it for SR requirements.
    """
    def __init__(self, channels):
        nn.Module.__init__(self)
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # Use PReLU instead of ReLU for better gradient flow in generator
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        return x + res


class Generator(nn.Module):
    def __init__(self, scale_factor=4):
        super(Generator, self).__init__()
        self.scale_factor = scale_factor
        upsample_num = int(math.log(scale_factor, 2))

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU()
        )
        self.res_blocks = nn.Sequential(
            SRResidualBlock(64),
            SRResidualBlock(64),
            SRResidualBlock(64),
            SRResidualBlock(64),
            SRResidualBlock(64),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )

        upsample_blocks = [UpsampleBlock(64, 64) for _ in range(upsample_num)]
        upsample_blocks.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.upsample_blocks = nn.Sequential(*upsample_blocks)

    def forward(self, x):
        block1 = self.block1(x)
        res_blocks = self.res_blocks(block1)
        final = self.final(res_blocks)
        output = block1 + final
        output = self.upsample_blocks(output)
        return (torch.tanh(output) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def forward(self, x):
        batch = x.size(0)
        return torch.sigmoid(self.disc(x).view(batch))