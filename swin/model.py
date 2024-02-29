import torch
import numpy
import torch.nn as nn
import torch.nn.functional as functional
import math
from einops import rearrange

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device

class Embedding(nn.Module):
    def __init__(self, patch_size=4, C=96):
        super().__init__()
        self.linear = nn.Conv2d(3, C, kernel_size=patch_size, stride=patch_size)
        self.layer_norm = nn.LayerNorm(C)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        x = self.relu(self.layer_norm(x))
        return x
    
class Merge(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.linear = nn.Linear(4*C, 2*C)
        self.norm = nn.LayerNorm(2*C)

    def forward(self, x):
        height = width = int(math.sqrt(x.shape[1]) / 2)
        x = rearrange(x, 'b (h s1 w s2) c -> b (h w) (s2 s1 c)', s1=2, s2=2, h=height, w=width)
        x = self.linear(x)
        x = self.norm(x)
        return x
    
class ShiftedWindowMSA(nn.Module):
    def __init__(self, embed_dim, n_heads, window_size, mask=False):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.window_size = window_size
        self.mask = mask
        self.proj1 = nn.Linear(embed_dim, 3*embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.embeddings = RelativeEmbedding()

    def forward(self, x):
        h_dim = self.embed_dim / self.n_heads
        height = width = int(math.sqrt(x.shape[1]))
        x = self.proj1(x)
        x = rearrange(x, 'b (h w) (c K) -> b h w c K', K=3, h=height, w=width)

        if self.mask:
            x = torch.roll(x, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2))

        x = rearrange(x, 'b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K',
                      H = self.n_heads, m1 = self.window_size, m2 = self.window_size)

        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        att_scores = (Q @ K.transpose(4, 5)) / math.sqrt(h_dim)
        att_scores = self.embeddings(att_scores)

        if self.mask:
            row_mask = torch.zeros((self.window_size**2, self.window_size**2)).cuda()
            row_mask[-self.window_size * (self.window_size//2):, 0:-self.window_size * (self.window_size//2)] = float('-inf')
            row_mask[0:-self.window_size * (self.window_size//2), -self.window_size * (self.window_size//2):] = float('-inf')
            column_mask = rearrange(row_mask, '(r w1) (c w2) -> (w1 r) (w2 c)', w1 = self.window_size, w2 = self.window_size).cuda()
            att_scores[:, :, -1, :] += row_mask
            att_scores[:, :, :, -1] += column_mask

        att = functional.softmax(att_scores, dim=-1) @ V
        x = rearrange(att, 'b H h w (m1 m2) E -> b (h m1) (w m2) (H E)', m1 = self.window_size, m2=self.window_size)

        if self.mask:
            x = torch.roll(x, (self.window_size//2, self.window_size//2), (1, 2))
        x = rearrange(x, 'b h w c -> b (h w) c')
        return self.proj2(x)

class RelativeEmbedding(nn.Module):
    def __init__(self, window_size=7):
        super().__init__()
        b = nn.Parameter(torch.randn(2*window_size-1, 2*window_size-1))
        x = torch.arange(1, window_size+1, 1/window_size)
        x = (x[None, :]-x[:, None]).int()
        y = torch.concat([torch.arange(1, window_size+1)] * window_size)
        y = (y[None, :]-y[:, None])
        self.embeddings = nn.Parameter((b[x[:,:], y[:,:]]), requires_grad=False)

    def forward(self, x):
        return x + self.embeddings
    
class SwinBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, mask):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.wmsa = ShiftedWindowMSA(embed_dim=embed_dim, n_heads=num_heads, window_size=window_size, mask=mask)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim*4),
            nn.GELU(),
            nn.Linear(embed_dim*4, embed_dim)
        )

    def forward(self, x):
        height, width = x.shape[1:3]
        res1 = self.dropout(self.wmsa(self.layer_norm(x)) + x)
        x = self.layer_norm(res1)
        x = self.mlp(x)
        return self.dropout(x + res1)

class AlternateSwin(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size=7):
        super().__init__()
        self.wsa = SwinBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=False)
        self.wmsa = SwinBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=True)

    def forward(self, x):
        return self.wmsa(self.wsa(x))

class SwinTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = Embedding()
        self.embedding = Embedding()
        self.patch1 = Merge(96)
        self.patch2 = Merge(192)
        self.patch3 = Merge(384)
        self.stage1 = AlternateSwin(96, 3)
        self.stage2 = AlternateSwin(192, 6)
        self.stage3_1 = AlternateSwin(384, 12)
        self.stage3_2 = AlternateSwin(384, 12)
        self.stage3_3 = AlternateSwin(384, 12)
        self.stage4 = AlternateSwin(768, 24)

    def forward(self, x):
        x = self.embedding(x)
        x = self.patch1(self.stage1(x))
        x = self.patch2(self.stage2(x))
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        x = self.stage3_3(x)
        x = self.patch3(x)
        x = self.stage4(x)
        return x