from torch import nn 
from refrakt_core.utils.classes.attention import MHA
from refrakt_core.utils.classes.resnet import SkipConnections
from refrakt_core.utils.classes.utils import FeedForward, LayerNormalization

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