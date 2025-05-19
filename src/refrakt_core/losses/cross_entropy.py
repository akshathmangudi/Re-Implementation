import torch.nn as nn
from refrakt_core.losses.templates.base import BaseLoss

class CrossEntropyLoss(BaseLoss):
    """
    Cross-Entropy Loss with optional label smoothing.
    """
    def __init__(self, weight=None, label_smoothing=0.0, device="cuda"):
        super().__init__(name="CrossEntropyLoss")
        self.loss = nn.CrossEntropyLoss(weight=weight, label_smoothing=label_smoothing)
        self.weight = weight
        self.label_smoothing = label_smoothing
        self.device = device

    def forward(self, pred, target):
        """
        Compute cross-entropy loss.
        Args:
            pred (torch.Tensor): Predictions of shape (N, C).
            target (torch.Tensor): Ground truth labels of shape (N,).
        Returns:
            torch.Tensor: Cross-entropy loss.
        """
        if pred.ndim != 2:
            raise ValueError(f"Expected pred to have shape (N, C), got {pred.shape}")
        if target.ndim != 1:
            raise ValueError(f"Expected target to have shape (N,), got {target.shape}")
        if pred.shape[0] != target.shape[0]:
            raise ValueError(f"Batch size mismatch: {pred.shape[0]} vs {target.shape[0]}")
        return self.loss(pred, target)


    def get_config(self):
        """
        Get the configuration of the loss function.
        Returns:
            dict: Configuration dictionary.
        """
        return {
            **super().get_config(),
            "weight": self.weight,
            "label_smoothing": self.label_smoothing,
            "device": self.device,
        }