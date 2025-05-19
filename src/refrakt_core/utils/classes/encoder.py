from torch import nn 
from refrakt_core.utils.classes.attention import MHA
from refrakt_core.utils.classes.resnet import SkipConnections
from refrakt_core.utils.classes.utils import FeedForward, LayerNormalization

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