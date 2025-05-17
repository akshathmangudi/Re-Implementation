import torch.nn as nn


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvNeXtBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size, stride, padding)
        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size, stride, padding)

        self.ln = nn.LayerNorm(out_channels)
        self.gelu = nn.GELU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.gelu(out)
        out = out.permute(0, 2, 3, 1)  # [batch, height, width, channels]
        out = self.ln(out)
        out = out.permute(0, 3, 1, 2)  # [batch, channels, height, width]

        out = self.conv2(out)
        return out


class ConvNeXt(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(ConvNeXt, self).__init__()
        self.stem = nn.Conv2d(
            in_channels, 96, kernel_size=4, stride=4)
        self.block1 = ConvNeXtBlock(96, 192)
        self.block2 = ConvNeXtBlock(192, 384)
        self.block3 = ConvNeXtBlock(384, 768)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
