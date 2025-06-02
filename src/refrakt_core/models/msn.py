# src/refrakt_core/models/msn.py

import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm import create_model

from refrakt_core.models.templates.base import BaseModel
from refrakt_core.registry.model_registry import register_model


@register_model("msn")
class MSNModel(BaseModel):
    def __init__(self, encoder_name, projector_dim, num_prototypes, pretrained=True):
        super().__init__()
        self.encoder = create_model(encoder_name, pretrained=pretrained, num_classes=0)
        self.target_encoder = create_model(
            encoder_name, pretrained=False, num_classes=0
        )

        # Freeze target encoder gradients
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        dim = projector_dim
        self.projector = nn.Sequential(
            nn.BatchNorm1d(self.encoder.num_features),
            nn.Linear(self.encoder.num_features, dim),
            nn.BatchNorm1d(dim),
            nn.ReLU(inplace=True),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim, affine=False),
        )

        self.target_projector = copy.deepcopy(self.projector)
        for p in self.target_projector.parameters():
            p.requires_grad = False

        self.prototypes = nn.Parameter(torch.randn(num_prototypes, dim))
        nn.init.xavier_uniform_(self.prototypes)

    def forward(self, x_anchor, x_target):
        """
        Args:
            x_anchor: masked view (B, C, H, W)
            x_target: unmasked view (B, C, H, W)
        Returns:
            z_anchor, z_target, prototypes
        """
        z_anchor = self.encoder(x_anchor)  # (B, D)
        z_anchor = self.projector(z_anchor)  # (B, D)
        z_anchor = F.normalize(z_anchor, dim=-1)

        with torch.no_grad():
            z_target = self.target_encoder(x_target)
            z_target = self.target_projector(z_target)
            z_target = F.normalize(z_target, dim=-1)

        return z_anchor, z_target, self.prototypes
