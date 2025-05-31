from tqdm import tqdm
import torch
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.registry.trainer_registry import register_trainer

@register_trainer("gan")
class GANTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,  # Dictionary of loss functions
        optimizer,  # Dictionary of optimizers
        device="cuda",
        scheduler=None,
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, device)
        
        # Ensure both loss and optimizer are dicts
        if not isinstance(loss_fn, dict) or not {"generator", "discriminator"}.issubset(loss_fn.keys()):
            raise ValueError("loss_fn must be a dictionary with 'generator' and 'discriminator' keys")
        
        if not isinstance(optimizer, dict) or not {"generator", "discriminator"}.issubset(optimizer.keys()):
            raise ValueError("optimizer must be a dictionary with 'generator' and 'discriminator' keys")

        self.loss_fns = loss_fn
        self.optimizers = optimizer
        self.extra_params = kwargs

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for batch in loop:
                # Move batch to device
                if isinstance(batch, dict):
                    device_batch = {k: v.to(self.device) for k, v in batch.items()}
                else:
                    device_batch = [tensor.to(self.device) for tensor in batch]
                
                # Use the model's training_step method if available
                losses = self.model.training_step(
                    device_batch, 
                    optimizer=self.optimizers, 
                    loss_fn=self.loss_fns, 
                    device=self.device
                )
                
                loop.set_postfix({
                    "gen_loss": losses.get('g_loss', 0),
                    "disc_loss": losses.get('d_loss', 0)
                })

    def evaluate(self):
        self.model.eval()
        total_psnr = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", leave=False):
                if isinstance(batch, dict):
                    lr = batch.get("lr", batch.get("input"))
                    hr = batch.get("hr", batch.get("target"))
                else:
                    lr, hr = batch[0], batch[1]
                
                lr = lr.to(self.device)
                hr = hr.to(self.device)
                sr = self.model.generate(lr)
                
                # Calculate PSNR
                mse = torch.mean((sr - hr) ** 2)
                psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
                total_psnr += psnr.item()
        
        avg_psnr = total_psnr / len(self.val_loader)
        print(f"\nValidation PSNR: {avg_psnr:.2f} dB")
        return avg_psnr