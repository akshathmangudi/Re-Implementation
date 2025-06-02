import torch
from tqdm import tqdm

from refrakt_core.registry.trainer_registry import register_trainer
from refrakt_core.trainer.base import BaseTrainer


@register_trainer("contrastive")
class ContrastiveTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        loss_fn,
        optimizer_cls,
        optimizer_args=None,
        device="cuda",
        scheduler=None,  # Add scheduler parameter
        **kwargs,  # Add kwargs for future compatibility
    ):
        super().__init__(model, train_loader, val_loader, device, **kwargs)
        self.loss_fn = loss_fn
        self.scheduler = scheduler  # Store scheduler
        self.extra_params = kwargs  # Store additional parameters

        if optimizer_args is None:
            optimizer_args = {"lr": 1e-3}

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)

    def train(self, num_epochs):
        best_accuracy = 0.0
        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0.0

            for batch in loop:
                # Handle different batch formats
                if isinstance(batch, (list, tuple)):
                    # Unpack views directly
                    view1, view2 = batch
                elif isinstance(batch, dict):
                    # Support dictionary format if needed
                    view1 = batch["view1"]
                    view2 = batch["view2"]
                else:
                    raise TypeError("Unsupported batch format for contrastive learning")

                view1 = view1.to(self.device)
                view2 = view2.to(self.device)

                self.optimizer.zero_grad()

                # Forward pass - both views through same model
                z1 = self.model(view1)
                z2 = self.model(view2)

                # Compute contrastive loss
                loss = self.loss_fn(z1, z2)
                loss.backward()
                self.optimizer.step()

                # Update progress and accumulate loss
                loop.set_postfix({"loss": loss.item()})
                epoch_loss += loss.item()

            # Step scheduler at epoch end if available
            if self.scheduler:
                self.scheduler.step()
                current_lr = self.optimizer.param_groups[0]["lr"]
                print(f"Epoch {epoch+1}: learning rate = {current_lr:.6f}")

            current_accuracy = self.evaluate()
            if current_accuracy > best_accuracy:
                best_accuracy = current_accuracy
                self.save(suffix="best_model")
                print(f"New best model saved with accuracy: {best_accuracy * 100:.2f}%")

            # Always save the latest model
            self.save(suffix="latest")

            # Log epoch-level metrics
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            print(f"Epoch {epoch+1} Train Loss: {avg_epoch_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        num_batches = len(self.val_loader)

        with torch.no_grad():
            loop = tqdm(self.val_loader, desc="Validating", leave=False)
            for batch in loop:
                # Handle batch formats same as training
                if isinstance(batch, (list, tuple)):
                    view1, view2 = batch
                elif isinstance(batch, dict):
                    view1 = batch["view1"]
                    view2 = batch["view2"]
                else:
                    raise TypeError("Unsupported batch format for contrastive learning")

                view1 = view1.to(self.device)
                view2 = view2.to(self.device)

                # Forward pass
                z1 = self.model(view1)
                z2 = self.model(view2)

                # Compute loss
                loss = self.loss_fn(z1, z2)
                total_loss += loss.item()

                loop.set_postfix({"loss": loss.item()})

        avg_val_loss = total_loss / num_batches
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
