import os 
import sys 
sys.path.append(os.path.abspath("../"))  # Add parent directory to Python path

import torch
from tqdm import tqdm
from refrakt import models
from refrakt.registry.model_registry import get_model

class Trainer:
    def __init__(self, model_name, train_loader, val_loader, 
                 loss_fn, optimizer, model_args={}, device="cuda"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.model = get_model(model_name, **model_args).to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def _default_training_step(self, batch):
        """Fallback for supervised models"""
        x, y = batch
        x, y = x.to(self.device), y.to(self.device)
        self.optimizer.zero_grad()
        outputs = self.model(x)
        loss = self.loss_fn(outputs, y)
        loss.backward()
        self.optimizer.step()
        return {"loss": loss.item()}

    def train(self, num_epochs=10):
        self.model.train()
        for epoch in range(num_epochs):
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            epoch_loss = 0.0
            for batch in loop:
                # Use model-specific training_step if defined
                if hasattr(self.model, "training_step"):
                    metrics = self.model.training_step(
                        batch, self.optimizer, self.loss_fn, self.device
                    )
                else:
                    metrics = self._default_training_step(batch)
                epoch_loss += metrics.get("loss", 0)
                loop.set_postfix(**metrics)
            print(f"Epoch {epoch+1} Loss: {epoch_loss/len(self.train_loader):.4f}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating"):
                if hasattr(self.model, "validation_step"):
                    metrics = self.model.validation_step(
                        batch, self.loss_fn, self.device
                    )
                else:
                    x, y = batch
                    x, y = x.to(self.device), y.to(self.device)
                    outputs = self.model(x)
                    loss = self.loss_fn(outputs, y)
                    metrics = {"val_loss": loss.item()}
                total_loss += metrics["val_loss"]
        print(f"Validation Loss: {total_loss/len(self.val_loader):.4f}")