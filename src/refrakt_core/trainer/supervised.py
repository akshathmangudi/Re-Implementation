import torch
from tqdm import tqdm
from refrakt_core.trainer.base import BaseTrainer

class SupervisedTrainer(BaseTrainer):
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
            optimizer_args = {"lr": 1e-4}
        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in loop:
                if isinstance(batch, (tuple, list)):
                    inputs, targets = batch
                elif isinstance(batch, dict):
                    inputs, targets = batch["input"], batch["target"]
                else:
                    raise TypeError("Unsupported batch format")

                inputs, targets = inputs.to(self.device), targets.to(self.device)

                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.loss_fn(outputs, targets)
                loss.backward()
                self.optimizer.step()
                loop.set_postfix({"loss": loss.item()})


    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                preds = self.model(batch["input"])
                # TODO: Add metric computations here
