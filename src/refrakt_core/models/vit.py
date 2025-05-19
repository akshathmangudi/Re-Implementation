import torch
import torch.nn as nn
from typing import Tuple
from refrakt_core.utils.classes.resnet import ViTResidual
from refrakt_core.models.templates.models import BaseClassifier
from refrakt_core.utils.methods import positional_embeddings, patchify
from refrakt_core.registry.model_registry import register_model

@register_model("vit")
class VisionTransformer(BaseClassifier):
    def __init__(
        self,
        image_size=224,
        patch_size=16,
        num_classes=1000,
        dim=768,
        depth=12,
        heads=12,
        in_channels=3,
        model_name="vit_classifier",
    ):
        super(VisionTransformer, self).__init__(
            num_classes=num_classes, model_name=model_name
        )
        
        assert image_size % patch_size == 0, "Image size must be divisible by patch size"
        self.n_patches = image_size // patch_size
        self.patch_size = patch_size
        self.hidden_d = dim
        
        self.input_d = in_channels * patch_size * patch_size
        
        self.linear_mapper = nn.Linear(self.input_d, dim)
        self.v_class = nn.Parameter(torch.rand(1, dim))
        self.register_buffer(
            "positional_embeddings",
            positional_embeddings(self.n_patches**2 + 1, dim),
            persistent=False,
        )

        self.blocks = nn.ModuleList(
            [ViTResidual(dim, heads) for _ in range(depth)]
        )

        self.mlp_head = nn.Sequential(nn.Linear(dim, num_classes))

    def forward_features(self, images: torch.Tensor) -> torch.Tensor:
        n = images.shape[0]
        patches = patchify(images, self.n_patches).to(self.positional_embeddings.device)

        tokens = self.linear_mapper(patches)
        tokens = torch.cat([self.v_class.expand(n, 1, -1), tokens], dim=1)
        x = tokens + self.positional_embeddings.repeat(n, 1, 1)

        for block in self.blocks:
            x = block(x)
        return x[:, 0]  # CLS token

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        cls_token = self.forward_features(images)
        return self.mlp_head(cls_token)
    
    def features(self, images: torch.Tensor) -> torch.Tensor:
        return self.forward_features(images)