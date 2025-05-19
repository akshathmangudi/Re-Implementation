import torch.nn as nn
import torch
from refrakt_core.registry.model_registry import register_model
from refrakt_core.utils.classes.embedding import Embedding
from refrakt_core.utils.classes.swin import AlternateSwin
from refrakt_core.utils.classes.utils import Merge

@register_model("swin")
class SwinTransformer(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
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

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = self.patch1(self.stage1(x))
        x = self.patch2(self.stage2(x))
        x = self.stage3_1(x)
        x = self.stage3_2(x)
        x = self.stage3_3(x)
        x = self.patch3(x)
        x = self.stage4(x)

        x = x.mean(dim=1)
        x = self.head(x)
        return x

