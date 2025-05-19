import torch.nn as nn
from torchvision.models import vgg19
from refrakt_core.losses.templates.base import BaseLoss

class PerceptualLoss(BaseLoss):
    """
    Perceptual Loss using a pre-trained VGG19 network.
    Computes the MSE loss between the feature maps of the predicted and target images.
    """
    def __init__(self, device="cuda"):
        super().__init__(name="PerceptualLoss")
        vgg = vgg19(pretrained=True).features[:36].to(device).eval()
        for param in vgg.parameters():
            param.requires_grad = False
        self.vgg = vgg
        self.device = device
        self.freeze()

    def forward(self, sr, hr):
        """
        Forward pass to compute perceptual loss.
        Args:
            sr (torch.Tensor): Super-resolved image of shape (N, C, H, W).
            hr (torch.Tensor): High-resolution target image of shape (N, C, H, W).
        Returns:
            torch.Tensor: Perceptual loss (MSE between feature maps).
        """
        sr_features = self.vgg(sr)
        hr_features = self.vgg(hr)
        return nn.functional.mse_loss(sr_features, hr_features)

    def get_config(self):
        """
        Get the configuration of the loss function.
        Returns:
            dict: Configuration dictionary.
        """
        return {
            **super().get_config(),
            "backbone": "vgg19",
            "layers_used": "features[:36]",
            "device": self.device,
        }