import math
import torch
from torch import nn 
from einops import rearrange
from torch.nn import functional
from refrakt_core.utils.classes.embedding import RelativeEmbedding

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
