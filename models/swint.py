import torch.nn as nn
from utils.classes import Embedding, Merge, AlternateSwin


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
