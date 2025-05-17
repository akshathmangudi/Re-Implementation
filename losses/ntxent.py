import torch
import torch.nn as nn


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        N = z1.size(0)
        if N <= 1:
            raise ValueError("Batch size must be > 1 for NT-Xent loss.")

        z = torch.cat([z1, z2], dim=0)
        sim_matrix = torch.matmul(z, z.T) / self.temperature
        mask = torch.eye(2 * N, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)

        positives = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
        pos_sim = sim_matrix[torch.arange(2 * N), positives]

        exp_sim = torch.exp(sim_matrix)
        loss = -torch.log(torch.exp(pos_sim) / exp_sim.sum(dim=1))
        return loss.mean()
