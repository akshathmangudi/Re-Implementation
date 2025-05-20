from tqdm import tqdm
import torch
from refrakt_core.trainer.base import BaseTrainer

class AETrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer_cls,
        optimizer_args=None,
        device="cuda"
    ):
        super().__init__(model, train_loader, val_loader, device)
        self.loss_fn = loss_fn
        if optimizer_args is None:
            optimizer_args = {"lr": 1e-3}
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in loop:
                metrics = self.model.training_step(
                    batch=batch,
                    optimizer=self.optimizer,
                    loss_fn=self.loss_fn,
                    device=self.device
                )
                loop.set_postfix(metrics)

    def evaluate(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in self.val_loader:
                metrics = self.model.validation_step(
                    batch=batch,
                    loss_fn=self.loss_fn,
                    device=self.device
                )
                val_loss += metrics["val_loss"]

        avg_val_loss = val_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
