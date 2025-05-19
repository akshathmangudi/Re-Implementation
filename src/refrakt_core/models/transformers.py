import math
import torch
import torch.nn as nn
from refrakt_core.utils.classes.encoder import Encoder
from refrakt_core.utils.classes.decoder import Decoder
from refrakt_core.utils.classes.transformers import InputEmbeddings, PositionalEncoding
from refrakt_core.utils.classes.utils import Projection


class Transformer(nn.Module):
    def __init__(
        self,
        encoder: Encoder,
        decoder: Decoder,
        src_embed: InputEmbeddings,
        tgt_embed: InputEmbeddings,
        src_pos: PositionalEncoding,
        tgt_pos: PositionalEncoding,
        proj: Projection,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.src_pos = src_pos
        self.tgt_post = tgt_pos
        self.proj = proj

    def encode(self, src, src_mask):
        src = self.src_embed(src)
        src = self.src_pos(src)
        return self.encoder(src, src_mask)

    def decoder(self, enc_output, src_mask, tgt, tgt_mask):
        tgt = self.tgt_embed(tgt)
        tgt = self.tgt_pos(tgt)
        return self.decoder(tgt, enc_output, src_mask, tgt_mask)

    def project(self, x):
        return self.proj(x)
