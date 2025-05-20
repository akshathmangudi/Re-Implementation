from tqdm import tqdm
import torch
from refrakt_core.trainer.base import BaseTrainer  # Adjust path accordingly

class GANTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fns,
        optimizers,
        device="cuda"
    ):
        super().__init__(model, train_loader, val_loader, device)

        # Ensure both loss and optimizer are dicts
        assert isinstance(loss_fns, dict) and "generator" in loss_fns and "discriminator" in loss_fns
        assert isinstance(optimizers, dict) and "generator" in optimizers and "discriminator" in optimizers

        self.loss_fns = loss_fns
        self.optimizers = optimizers

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in loop:
                metrics = self.model.training_step(
                    batch=batch,
                    optimizer=self.optimizers,
                    loss_fn=self.loss_fns,
                    device=self.device
                )
                loop.set_postfix(metrics)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                lr, _ = batch
                lr = lr.to(self.device)
                sr = self.model.generate(lr)
                # Optional: compute PSNR, SSIM, or save generated images
