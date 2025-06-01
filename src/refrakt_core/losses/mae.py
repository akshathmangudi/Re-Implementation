import torch

from refrakt_core.losses.templates.base import \
    BaseLoss  # adjust import path if needed
from refrakt_core.registry.loss_registry import register_loss


@register_loss("mae")
class MAELoss(BaseLoss):
    """
    Masked Autoencoder Loss.
    Computes MSE only over masked patches.
    Supports optional normalization over patches (MAE paper).
    """

    def __init__(self, normalize_target=False):
        super().__init__(name="MAELoss")
        self.normalize_target = normalize_target

    def forward(self, predictions, targets=None):
        """
        Args:
            predictions (dict):
                - recon_patches: (B, N, patch_dim)
                - mask: (B, N)
                - original_patches: (B, N, patch_dim)
            targets (unused): For compatibility; ignored.
        Returns:
            torch.Tensor: scalar masked reconstruction loss
        """
        pred = predictions["recon_patches"]
        mask = predictions["mask"].unsqueeze(-1)
        original = predictions["original_patches"]

        if self.normalize_target:
            mean = original.mean(dim=-1, keepdim=True)
            std = original.std(dim=-1, keepdim=True) + 1e-6
            original = (original - mean) / std

        loss = ((pred - original) ** 2) * mask
        return loss.sum() / mask.sum()

    def get_config(self):
        config = super().get_config()
        config.update({
            "normalize_target": self.normalize_target
        })
        return config

    def extra_repr(self):
        return f"name={self.name}, normalize_target={self.normalize_target}"
