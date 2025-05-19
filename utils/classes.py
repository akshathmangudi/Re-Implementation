import math
import torch
import torch.nn as nn
from pathlib import Path
from PIL import Image
from einops import rearrange
from typing import Tuple
from torch.nn import functional
from utils.methods import find_classes
from torch.utils.data import Dataset

class MSA(nn.Module):
    def __init__(self, d, n_heads=4):
        super(MSA, self).__init__()
        assert d % n_heads == 0
        d_head = d // n_heads
        self.q_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(n_heads)]
        )
        self.k_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(n_heads)]
        )
        self.v_mappings = nn.ModuleList(
            [nn.Linear(d_head, d_head) for _ in range(n_heads)]
        )
        self.d = d
        self.n_heads = n_heads
        self.d_head = d_head
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, sequences):
        result = []
        for sequence in sequences:
            seq_result = []
            for head in range(self.n_heads):
                q = self.q_mappings[head](
                    sequence[:, head * self.d_head : (head + 1) * self.d_head]
                )
                k = self.k_mappings[head](
                    sequence[:, head * self.d_head : (head + 1) * self.d_head]
                )
                v = self.v_mappings[head](
                    sequence[:, head * self.d_head : (head + 1) * self.d_head]
                )
                attention = self.softmax(q @ k.T / (self.d_head**0.5))
                seq_result.append(attention @ v)
            result.append(torch.hstack(seq_result))
        return torch.stack(result)


class Residual(nn.Module):
    def __init__(self, hidden_d, n_heads, mlp_ratio=4):
        super(Residual, self).__init__()
        self.norm1 = nn.LayerNorm(hidden_d)
        self.mhsa = MSA(hidden_d, n_heads)
        self.norm2 = nn.LayerNorm(hidden_d)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_d, mlp_ratio * hidden_d),
            nn.GELU(),
            nn.Linear(mlp_ratio * hidden_d, hidden_d),
        )

    def forward(self, x):
        out = x + self.mhsa(self.norm1(x))
        out = out + self.mlp(self.norm2(out))
        return out


class InputEmbeddings(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embedding(x) * math.sqrt(self.d_model)


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout: float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * denom)
        pe[:, 1::2] = torch.cos(position * denom)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + (self.pe[:, : x.shape[1], :]).requires_grad_(
            False
        )  # Not a learnable parameter
        return self.dropout(x)


class LayerNormalization(nn.Module):
    def __init__(self, eps: float = 10**-6) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.alpha * (x - mean) / (std + self.eps) + self.bias


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout(torch.relu(x))
        x = self.linear_2(x)
        return x


class MHA(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        # Create Q, K, V matrix.
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

        assert d_model % n_heads == 0, "d_model is not divisible by n_heads"

        self.d_k = d_model // n_heads

        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)

        self.w_o = nn.Linear(d_model, d_model)

    @staticmethod
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]

        att_scores = (query @ key.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            att_scores.masked_fill_(mask == 0, -1e10)
        att_scores = att_scores.softmax(dim=-1)  # (batch, h, seq_len, seq_len)
        if dropout is not None:
            att_scores = dropout(att_scores)
        return (att_scores @ value), att_scores

    def forward(self, q, k, v, mask):
        query = self.w_q(q)
        key = self.w_k(k)
        value = self.w_v(v)

        # (batch, seq_len, d_model) --> (batch, seq_len, h, d_k) --> (batch, h, seq_len, d_k)
        query = query.view(query.shape[0], query.shape[1], self.h, self.d_k).transpose(
            1, 2
        )
        key = key.view(key.shape[0], key.shape[1], self.h, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.h, self.d_k).transpose(
            1, 2
        )

        x, self.att_scores = MHA.attention(query, key, value, mask, self.dropout)

        # (batch, h, seq_len, d_k) --> (batch, seq_len, h, d_k)
        x = x.transpose(1, 2).contiguous().view(x.shape[0], -1, self.h * self.d_k)
        return self.w_o(x)


class SkipConnections(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.norm = LayerNormalization()

    def forward(self, x, sub):
        y = self.norm(x)
        y = sublayer(y)
        y = self.dropout(y)
        return x + y


class EncoderBlock(nn.Module):
    def __init__(self, self_att: MHA, feed_forw: FeedForward, dropout: float) -> None:
        super().__init__()
        self.self_att = self_att
        self.feed_forw = feed_forw
        self.dropout = nn.Dropout(dropout)
        self.skip_conn = nn.ModuleList([SkipConnections(dropout) for _ in range(2)])

    def forward(self, x, src_mask):
        x = self.skip_conn[0](x, lambda x: self.self_att(x, x, x, src_mask))
        x = self.skip_conn[1](x, self.feed_forw)
        return x


class Encoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(
        self, masked_att: MHA, cross_att: MHA, feed_forw: FeedForward, dropout: float
    ) -> None:
        super().__init__()
        self.masked_att = masked_att
        self.cross_att = cross_att
        self.feed_forw = feed_forw
        self.dropout = nn.Dropout(dropout)
        self.skip_conn = nn.ModuleList([SkipConnections(dropout) for _ in range(3)])

    def forward(self, x, enc_output, src_mask, tgt_mask):
        x = self.skip_conn[0](x, lambda x: self.masked_att(x, x, x, tgt_mask))
        x = self.skip_conn[1](
            x, lambda x: self.cross_att(x, enc_output, enc_output, src_mask)
        )
        x = self.skip_conn[2](x, self.feed_forw)
        return x


class Decoder(nn.Module):
    def __init__(self, layers: nn.ModuleList) -> None:
        super().__init__()
        self.layers = layers
        self.norm = LayerNormalization()

    def forward(self, x, enc_output, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, enc_output, src_mask, tgt_mask)
        return self.norm(x)


class Projection(nn.Module):
    def __init__(self, d_model: int, vocab_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        return torch.log_softmax(self.proj(x), dim=-1)


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


class Merge(nn.Module):
    def __init__(self, C):
        super().__init__()
        self.linear = nn.Linear(4 * C, 2 * C)
        self.norm = nn.LayerNorm(2 * C)

    def forward(self, x):
        height = width = int(math.sqrt(x.shape[1]) / 2)
        x = rearrange(
            x, "b (h s1 w s2) c -> b (h w) (s2 s1 c)", s1=2, s2=2, h=height, w=width
        )
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
        self.proj1 = nn.Linear(embed_dim, 3 * embed_dim)
        self.proj2 = nn.Linear(embed_dim, embed_dim)
        self.embeddings = RelativeEmbedding()
    
    def forward(self, x):
        # Get the device from input tensor
        device = x.device
        
        h_dim = self.embed_dim / self.n_heads
        height = width = int(math.sqrt(x.shape[1]))
        
        x = self.proj1(x)
        x = rearrange(x, "b (h w) (c K) -> b h w c K", K=3, h=height, w=width)
        
        if self.mask:
            x = torch.roll(
                x, (-self.window_size // 2, -self.window_size // 2), dims=(1, 2)
            )
            
        x = rearrange(
            x, "b (h m1) (w m2) (H E) K -> b H h w (m1 m2) E K",
            H=self.n_heads, m1=self.window_size, m2=self.window_size,
        )
        
        Q, K, V = x.chunk(3, dim=6)
        Q, K, V = Q.squeeze(-1), K.squeeze(-1), V.squeeze(-1)
        
        att_scores = (Q @ K.transpose(4, 5)) / math.sqrt(h_dim)
        att_scores = self.embeddings(att_scores)
        
        if self.mask:
            # Create masks on the appropriate device
            row_mask = torch.zeros((self.window_size**2, self.window_size**2), device=device)
            row_mask[
                -self.window_size * (self.window_size // 2):, 
                0:-self.window_size * (self.window_size // 2),
            ] = float("-inf")
            row_mask[
                0:-self.window_size * (self.window_size // 2), 
                -self.window_size * (self.window_size // 2):,
            ] = float("-inf")
            
            column_mask = rearrange(
                row_mask, "(r w1) (c w2) -> (w1 r) (w2 c)",
                w1=self.window_size, w2=self.window_size,
            )
            
            att_scores[:, :, -1, :] += row_mask
            att_scores[:, :, :, -1] += column_mask
        
        att = functional.softmax(att_scores, dim=-1) @ V
        
        x = rearrange(
            att, "b H h w (m1 m2) E -> b (h m1) (w m2) (H E)",
            m1=self.window_size, m2=self.window_size,
        )
        
        if self.mask:
            x = torch.roll(x, (self.window_size // 2, self.window_size // 2), (1, 2))
            
        x = rearrange(x, "b h w c -> b (h w) c")
        
        return self.proj2(x)


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

class CreateDataset(Dataset):
    """
    This dataset class was created as described by the documentation 
    provided by PyTorch. Most of the details here are explanatory. 
    """
    def __init__(self, target_dir: str, transform=None) -> None:
        self.paths = list(Path(target_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_dir)

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(image), class_idx
        else:
            return image, class_idx


class FlattenTransform:
    def __call__(self, x):
        return torch.flatten(x)
    
