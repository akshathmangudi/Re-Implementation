import torch.nn as nn
from torchvision.models import vgg19
from templates.base import BaseLoss


class PerceptualLoss(BaseLoss):
    def __init__(self, device="cuda"):
        super().__init__(name="PerceptualLoss")
        vgg = vgg19(pretrained=True).features[:36].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.freeze()

    def forward(self, sr, hr):
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return nn.functional.mse_loss(sr_features, hr_features)

    def get_config(self):
        return {
            **super().get_config(),
            "backbone": "vgg19",
            "layers_used": "features[:36]",
        }
