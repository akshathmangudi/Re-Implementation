import torch
import torch.nn as nn
from refrakt_core.losses.templates.base import BaseLoss


class NTXentLoss(BaseLoss):
    """
    NT-Xent (Normalized Temperature-scaled Cross Entropy) Loss
    
    This loss function is commonly used in contrastive learning frameworks like SimCLR.
    It pulls positive pairs together while pushing apart negative pairs in the embedding space.
    
    Parameters:
    -----------
    temperature : float, default=0.5
        Scaling factor for the similarity scores. Lower values make the model more
        sensitive to hard negatives.
    name : str, optional
        Name of the loss function. Defaults to the class name.
    """
    def __init__(self, temperature: float = 0.5, name: str = None):
        super().__init__(name=name)
        self.temperature = temperature
    
    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute the NT-Xent loss between two batches of embeddings.
        
        Parameters:
        -----------
        z1 : torch.Tensor
            First batch of embeddings with shape [N, D]
        z2 : torch.Tensor
            Second batch of embeddings with shape [N, D]
            
        Returns:
        --------
        torch.Tensor
            Scalar loss value
            
        Raises:
        -------
        ValueError
            If batch size is <= 1
        """
        N = z1.size(0)
        if N <= 1:
            raise ValueError("Batch size must be > 1 for NT-Xent loss.")
        
        z = torch.cat([z1, z2], dim=0)
        
        z_norm = nn.functional.normalize(z, dim=1)
        sim_matrix = torch.matmul(z_norm, z_norm.T) / self.temperature
        
        # Mask out self-comparisons
        mask = torch.eye(2 * N, device=z.device).bool()
        sim_matrix.masked_fill_(mask, -9e15)
        
        positives = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)]).to(z.device)
        pos_sim = sim_matrix[torch.arange(2 * N), positives]
        
        exp_sim = torch.exp(sim_matrix)
        loss = -torch.log(torch.exp(pos_sim) / exp_sim.sum(dim=1))
        
        return loss.mean()
    
    def get_config(self):
        """Return detailed configuration of the loss function."""
        config = super().get_config()
        config.update({
            "temperature": self.temperature,
            "type": "contrastive"
        })
        return config
    
    def extra_repr(self):
        """Additional information for string representation."""
        return f"name={self.name}, temperature={self.temperature}"