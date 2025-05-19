import torch
from tqdm import tqdm
from refrakt_core.registry.model_registry import get_model
from refrakt_core.models.templates.models import BaseGAN

class Trainer:
    def __init__(
        self,
        model_name,
        model_args,
        train_loader,
        val_loader,
        loss_fn,
        optimizer,
        device="cuda"
    ):
        self.device = torch.device(device)
        self.model = get_model(model_name, **model_args).to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

        self.is_gan = isinstance(self.model, BaseGAN)

        if self.is_gan:
            assert isinstance(loss_fn, dict), "`loss_fn` must be a dict for GANs"
            assert isinstance(optimizer, dict), "`optimizer` must be a dict for GANs"
            self.loss_fns = loss_fn
            self.optimizers = optimizer
        else:
            self.loss_fn = loss_fn
            self.optimizer = optimizer(self.model.parameters(), lr=1e-4)

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch in loop:
                if self.is_gan:
                    metrics = self.model.training_step(
                        batch=batch,
                        optimizer=self.optimizers,
                        loss_fn=self.loss_fns,
                        device=self.device
                    )
                else:
                    # Assume batch is a dict with "input" and "target" keys
                    batch = {k: v.to(self.device) for k, v in batch.items()}
                    self.optimizer.zero_grad()
                    outputs = self.model(batch["input"])
                    loss = self.loss_fn(outputs, batch["target"])
                    loss.backward()
                    self.optimizer.step()
                    metrics = {"loss": loss.item()}

                loop.set_postfix(metrics)

    def evaluate(self):
        self.model.eval()
        with torch.no_grad():
            for batch in self.val_loader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                if self.is_gan:
                    preds = self.model.generate(batch["lr"])
                else:
                    preds = self.model(batch["input"])
                # Optional: compute metrics (e.g., PSNR, SSIM)
