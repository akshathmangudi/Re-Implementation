from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from templates.models import BaseContrastiveModel


class SimCLRModel(BaseContrastiveModel):
    def __init__(self, proj_dim=128):
        super().__init__(
            model_name="simclr", backbone_name="resnet50", proj_dim=proj_dim
        )

        self.encoder = models.resnet50(pretrained=False)
        self.encoder.fc = nn.Identity()

        self.projector = nn.Sequential(
            nn.Linear(2048, 2048, bias=False),
            nn.ReLU(),
            nn.Linear(2048, proj_dim, bias=False),
        )

    def encode(self, x):
        return self.encoder(x)

    def project(self, h):
        return self.projector(h)

    def forward(self, x):
        h = self.encode(x)
        z = self.project(h)
        return F.normalize(z, dim=1)
