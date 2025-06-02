import torch
import torch.nn as nn
import torch.nn.functional as F

from refrakt_core.losses.templates.base import BaseLoss
from refrakt_core.registry.loss_registry import register_loss


@register_loss("dino")
class DINOLoss(BaseLoss):
    """
    DINO Loss implementation as described in the original paper.
    This loss computes the cross-entropy between student and teacher outputs
    using temperature scaling and centering mechanisms.
    """

    def __init__(
        self, out_dim=65536, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9
    ):
        super().__init__(name="DINOLoss")
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum
        self.register_buffer("center", torch.zeros(1, out_dim))

    def forward(self, student_output, teacher_output):
        """
        Compute the DINO loss.

        Args:
            student_output (torch.Tensor): Student's predictions (B, num_views, out_dim)
            teacher_output (torch.Tensor): Teacher's predictions (B, 1, out_dim)

        Returns:
            torch.Tensor: Scalar loss value
        """
        # Ensure inputs are float and on the same device
        device = student_output.device
        student_output = student_output.float().to(device)
        teacher_output = teacher_output.float().detach().to(device)

        # Get batch size and number of views
        n_views = student_output.shape[1]
        total_loss = 0.0
        n_loss_terms = 0

        # Center and sharpen teacher output
        teacher_probs = F.softmax(
            (teacher_output - self.center) / self.teacher_temp, dim=-1
        )

        for v in range(n_views):
            student_probs = F.log_softmax(
                student_output[:, v, :] / self.student_temp, dim=-1
            )
            loss = torch.sum(-teacher_probs * student_probs, dim=-1).mean()
            total_loss += loss
            n_loss_terms += 1

        self.update_center(teacher_output)
        return total_loss / n_loss_terms

    @torch.no_grad()
    def update_center(self, teacher_output):
        """
        Momentum update for the teacher's output center.
        """
        batch_center = torch.mean(teacher_output, dim=0, keepdim=True)
        self.center = self.center * self.center_momentum + batch_center * (
            1 - self.center_momentum
        )
