import torch.nn as nn

from refrakt_core.models.dino import DINOModel
from refrakt_core.models.resnet import ResNet18, ResNet50, ResNet101, ResNet152
from refrakt_core.registry.model_registry import register_model


class DINOBackboneWrapper(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.feature_dim = backbone.feature_dim

    def forward(self, x):
        return self.backbone(x, return_features=True)

@register_model("dino")
class DINOModelWrapper(DINOModel):
    def __init__(self, backbone="resnet18", out_dim=65536):
        backbone_map = {
            "resnet18": ResNet18,
            "resnet50": ResNet50,
            "resnet101": ResNet101,
            "resnet152": ResNet152,
        }

        if isinstance(backbone, str):
            if backbone not in backbone_map:
                raise ValueError(f"Unsupported backbone '{backbone}' for DINO.")
            backbone_instance = backbone_map[backbone]()
        elif isinstance(backbone, nn.Module):
            backbone_instance = backbone
        else:
            raise TypeError(f"Expected backbone to be str or nn.Module, got {type(backbone)}")

        wrapped = DINOBackboneWrapper(backbone_instance)
        super().__init__(backbone=wrapped, model_name="dino", out_dim=out_dim)
