# trainer/ae.py

from tqdm import tqdm
import torch
from refrakt_core.trainer.base import BaseTrainer
from refrakt_core.registry.trainer_registry import register_trainer


@register_trainer("autoencoder")
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
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                elif isinstance(batch, dict):
                    inputs = batch["image"]
                else:
                    inputs = batch

                inputs = inputs.to(self.device)

                self.optimizer.zero_grad()
                raw_outputs = self.model(inputs)

                # === Handle structured outputs for custom loss functions ===
                if isinstance(raw_outputs, dict):
                    loss = self.loss_fn(raw_outputs)
                elif isinstance(raw_outputs, tuple) and len(raw_outputs) == 3:
                    recon, mask, original = raw_outputs
                    structured_output = {
                        "recon_patches": recon,
                        "mask": mask,
                        "original_patches": original
                    }
                    loss = self.loss_fn(structured_output)
                else:
                    loss = self.loss_fn(raw_outputs, inputs)

                loss.backward()
                self.optimizer.step()

                loop.set_postfix({"loss": loss.item()})


    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)
            for batch in loop:
                if isinstance(batch, (list, tuple)):
                    inputs = batch[0]
                elif isinstance(batch, dict):
                    inputs = batch["image"]
                else:
                    inputs = batch

                inputs = inputs.to(self.device)
                raw_outputs = self.model(inputs)

                if isinstance(raw_outputs, dict):
                    loss = self.loss_fn(raw_outputs)
                elif isinstance(raw_outputs, tuple) and len(raw_outputs) == 3:
                    recon, mask, original = raw_outputs
                    structured_output = {
                        "recon_patches": recon,
                        "mask": mask,
                        "original_patches": original
                    }
                    loss = self.loss_fn(structured_output)
                else:
                    loss = self.loss_fn(raw_outputs, inputs)

                total_loss += loss.item()

        avg_val_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
