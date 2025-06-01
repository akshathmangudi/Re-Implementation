# src/refrakt_core/losses/msn.py

import torch
import torch.nn as nn
import torch.nn.functional as F

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss


@register_loss("msn")
class MSNLoss(BaseLoss):
    def __init__(self, temp_anchor=0.1, temp_target=0.04, lambda_me_max=1.0):
        super().__init__()
        self.temp_anchor = temp_anchor
        self.temp_target = temp_target
        self.lambda_me_max = lambda_me_max

    def forward(self, z_anchor, z_target, prototypes):
        """
        Args:
            z_anchor: (B*M, D)
            z_target: (B, D)
            prototypes: (K, D)
        """
        B = z_target.shape[0]
        M = z_anchor.shape[0] // B
        K = prototypes.shape[0]

        logits_anchor = torch.matmul(z_anchor, prototypes.T) / self.temp_anchor
        logits_target = torch.matmul(z_target, prototypes.T) / self.temp_target

        p_anchor = F.softmax(logits_anchor, dim=-1)           # (BM, K)
        p_target = F.softmax(logits_target, dim=-1).repeat_interleave(M, dim=0)  # (BM, K)

        loss_ce = F.cross_entropy(p_anchor.log(), p_target.detach(), reduction='none').mean()

        p_mean = p_anchor.mean(dim=0)
        entropy = -torch.sum(p_mean * torch.log(p_mean + 1e-6))
        loss_entropy = -self.lambda_me_max * entropy

        return loss_ce + loss_entropy
