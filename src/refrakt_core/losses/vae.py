import torch
import torch.nn as nn
import torch.nn.functional as F
from refrakt_core.registry.loss_registry import register_loss

@register_loss("vae")
class VAELoss(nn.Module):
    def __init__(self, recon_loss_type='mse', kld_weight=1.0):
        super().__init__()
        self.recon_loss_type = recon_loss_type
        self.kld_weight = kld_weight
        
    def forward(self, model_output, target):
        # Handle both dictionary and tensor outputs
        if isinstance(model_output, dict):
            recon = model_output['recon']
            mu = model_output['mu']
            logvar = model_output['logvar']
        else:
            # Assume simple autoencoder output
            recon = model_output
            mu, logvar = None, None
        
        # Reconstruction loss
        if self.recon_loss_type == 'mse':
            recon_loss = F.mse_loss(recon, target, reduction='sum')
        elif self.recon_loss_type == 'l1':
            recon_loss = F.l1_loss(recon, target, reduction='sum')
        else:
            raise ValueError(f"Unsupported reconstruction loss type: {self.recon_loss_type}")
        
        # Return only reconstruction loss for simple autoencoders
        if mu is None or logvar is None:
            return recon_loss
        
        # KL divergence loss for VAEs
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        return recon_loss + self.kld_weight * kld_loss