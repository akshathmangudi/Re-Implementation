from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from refrakt_core.registry.model_registry import register_model
from refrakt_core.models.templates.models import BaseContrastiveModel

@register_model("simclr")
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
        
    def training_step(self, batch, optimizer, loss_fn, device):
        """Contrastive training step"""
        x_i, x_j = batch
        x_i, x_j = x_i.to(device), x_j.to(device)
        optimizer.zero_grad()
        z_i, z_j = self(x_i), self(x_j)
        loss = loss_fn(z_i, z_j)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def validation_step(self, batch, loss_fn, device):
        x_i, x_j = batch
        x_i, x_j = x_i.to(device), x_j.to(device)
        z_i, z_j = self(x_i), self(x_j)
        loss = loss_fn(z_i, z_j)
        return {"val_loss": loss.item()}

    def encode(self, x):
        return self.encoder(x)

    def project(self, h):
        return self.projector(h)

    def forward(self, x):
        h = self.encode(x)
        z = self.project(h)
        return F.normalize(z, dim=1)
