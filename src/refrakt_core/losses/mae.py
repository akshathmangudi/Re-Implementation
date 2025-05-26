import torch
import torch.nn as nn

class MAELoss(nn.Module):
    """
    Custom loss for MAE that computes MSE over masked patches only.
    Optionally supports per-patch normalization (as described in the MAE paper).
    """
    def __init__(self, normalize_target=False):
        super().__init__()
        self.normalize_target = normalize_target

    def forward(self, predictions, targets=None):
        """
        Args:
            predictions (dict):
                - recon_patches: (B, N, patch_dim)
                - mask: (B, N)
                - original_patches: (B, N, patch_dim)
            targets (unused): Provided for API consistency, should be None.
        Returns:
            torch.Tensor: scalar masked reconstruction loss
        """
        pred = predictions["recon_patches"]          # (B, N, patch_dim)
        mask = predictions["mask"].unsqueeze(-1)     # (B, N, 1)
        original = predictions["original_patches"]   # (B, N, patch_dim)

        if self.normalize_target:
            mean = original.mean(dim=-1, keepdim=True)
            std = original.std(dim=-1, keepdim=True) + 1e-6
            original = (original - mean) / std

        loss = (pred - original) ** 2
        loss = (loss * mask).sum() / mask.sum()

        return loss
