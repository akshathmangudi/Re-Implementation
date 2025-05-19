import torch
import torch.nn as nn
from refrakt_core.losses.templates.base import BaseLoss

class GANLoss(BaseLoss):
    """
    GAN Loss for adversarial training.
    Supports both BCE and LSGAN loss.
    """
    def __init__(self, use_lsgan=False, device="cuda"):
        super().__init__(name="GANLoss")
        self.loss = nn.MSELoss() if use_lsgan else nn.BCEWithLogitsLoss()
        self.use_lsgan = use_lsgan
        self.device = torch.device(device)

    def forward(self, pred, target_is_real):
        """
        Compute GAN loss for discriminator or generator.
        Args:
            pred (torch.Tensor): Predictions from the discriminator.
            target_is_real (bool): True if the target is real, False if fake.
        Returns:
            torch.Tensor: GAN loss.
        """
        if not isinstance(target_is_real, bool):
            raise TypeError("target_is_real must be a boolean.")
        if not isinstance(pred, torch.Tensor):
            raise TypeError("pred must be a torch.Tensor.")

        target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
        target = target.to(pred.device)  # Ensure target is on the same device as pred

        return self.loss(pred, target)

    def get_config(self):
        """
        Get the configuration of the loss function.
        Returns:
            dict: Configuration dictionary.
        """
        return {
            **super().get_config(),
            "use_lsgan": self.use_lsgan,
            "device": str(self.device),
        }
