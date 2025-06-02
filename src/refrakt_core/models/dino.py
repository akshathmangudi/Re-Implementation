import torch
import torch.nn as nn
import torch.nn.functional as F

from refrakt_core.models.templates.base import BaseModel
from refrakt_core.registry.model_registry import register_model


class DINOHead(nn.Module):
    """
    Projection head for DINO as used in the paper.
    """

    def __init__(
        self, in_dim, out_dim=65536, hidden_dim=2048, bottleneck_dim=256, num_layers=3
    ):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(in_dim, hidden_dim))
            elif i < num_layers - 1:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, bottleneck_dim))
            if i < num_layers - 1:
                layers.append(nn.GELU())

        self.mlp = nn.Sequential(*layers)
        self.last_layer = nn.utils.weight_norm(
            nn.Linear(bottleneck_dim, out_dim, bias=False)
        )
        self.last_layer.weight_g.data.fill_(1.0)

    def forward(self, x):
        x = self.mlp(x)
        x = F.normalize(self.last_layer(x), dim=-1)
        return x


class DINOModel(BaseModel):
    """
    DINO model that returns multi-view projections for student and teacher networks.
    """

    def __init__(self, backbone, model_name="dino", out_dim=65536):
        super().__init__(model_name=model_name, model_type="contrastive")
        self.backbone = backbone  # must output flat features
        self.student_head = DINOHead(in_dim=backbone.feature_dim, out_dim=out_dim)
        self.teacher_head = DINOHead(in_dim=backbone.feature_dim, out_dim=out_dim)
        self.teacher_head.load_state_dict(self.student_head.state_dict())
        for param in self.teacher_head.parameters():
            param.requires_grad = False

    def forward(self, x, teacher=False):
        """
        Args:
            x (Tensor): Input tensor of shape (B, C, H, W)
            teacher (bool): If True, use teacher head

        Returns:
            Tensor: Projected features
        """
        features = self.backbone(x)
        if teacher:
            return self.teacher_head(features)
        return self.student_head(features)

    @torch.no_grad()
    def update_teacher(self, momentum=0.996):
        """
        EMA update for teacher parameters.
        """
        for student_param, teacher_param in zip(
            self.student_head.parameters(), self.teacher_head.parameters()
        ):
            teacher_param.data = (
                momentum * teacher_param.data + (1.0 - momentum) * student_param.data
            )
