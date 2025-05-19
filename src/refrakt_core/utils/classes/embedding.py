import torch
from torch import nn
from einops import rearrange

class Embedding(nn.Module):
    def __init__(self, patch_size=4, C=96):
        super().__init__()
        self.linear = nn.Conv2d(3, C, kernel_size=patch_size, stride=patch_size)
        self.layer_norm = nn.LayerNorm(C)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = rearrange(x, "b c h w -> b (h w) c")
        x = self.relu(self.layer_norm(x))
        return x


class RelativeEmbedding(nn.Module):
    def __init__(self, window_size=7):
        super().__init__()
        b = nn.Parameter(torch.randn(2 * window_size - 1, 2 * window_size - 1))
        x = torch.arange(1, window_size + 1, 1 / window_size)
        x = (x[None, :] - x[:, None]).int()
        y = torch.concat([torch.arange(1, window_size + 1)] * window_size)
        y = y[None, :] - y[:, None]
        self.embeddings = nn.Parameter((b[x[:, :], y[:, :]]), requires_grad=False)

    def forward(self, x):
        return x + self.embeddings