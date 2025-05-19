from torch import nn 
from refrakt_core.utils.classes.attention import ShiftedWindowMSA

class SwinBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, mask):
        super().__init__()
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.1)
        self.wmsa = ShiftedWindowMSA(
            embed_dim=embed_dim, n_heads=num_heads, window_size=window_size, mask=mask
        )
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
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
        self.wsa = SwinBlock(
            embed_dim=embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            mask=False,
        )
        self.wmsa = SwinBlock(
            embed_dim=embed_dim, num_heads=num_heads, window_size=window_size, mask=True
        )

    def forward(self, x):
        return self.wmsa(self.wsa(x))