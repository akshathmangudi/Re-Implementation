# vit_with_base.py
import torch
import torch.nn as nn
from typing import Tuple
from utils.classes import Residual
from templates.models import BaseClassifier
from utils.methods import positional_embeddings, patchify


class VisionTransformer(BaseClassifier):
    def __init__(
        self,
        chw: Tuple[int, int, int],
        num_classes: int,
        n_patches: int = 16,
        n_blocks: int = 2,
        hidden_d: int = 8,
        n_heads: int = 4,
        model_name: str = "vit_classifier",
    ):
        super(VisionTransformer, self).__init__(
            num_classes=num_classes, model_name=model_name
        )
        c, h, w = chw
        assert h % n_patches == 0 and w % n_patches == 0
        patch_size = (h // n_patches, w // n_patches)
        self.input_d = c * patch_size[0] * patch_size[1]
        self.linear_mapper = nn.Linear(self.input_d, hidden_d)
        self.v_class = nn.Parameter(torch.rand(1, hidden_d))
        self.register_buffer(
            "positional_embeddings",
            positional_embeddings(n_patches**2 + 1, hidden_d),
            persistent=False,
        )
        self.blocks = nn.ModuleList(
            [Residual(hidden_d, n_heads) for _ in range(n_blocks)]
        )
        self.mlp_head = nn.Sequential(nn.Linear(hidden_d, num_classes))
        self.n_patches = n_patches
        self.hidden_d = hidden_d

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        n = images.shape[0]
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)
        tokens = self.linear_mapper(patches)
        tokens = torch.cat([self.v_class.expand(n, 1, -1), tokens], dim=1)
        x = tokens + self.positional_embeddings.repeat(n, 1, 1)
        for block in self.blocks:
            x = block(x)
        return self.mlp_head(x[:, 0])
