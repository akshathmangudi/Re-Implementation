import torch
from torch import autocast
from torch.amp import GradScaler
from refrakt_core.trainer.base import BaseTrainer
import torch.nn.functional as F
from tqdm import tqdm
from refrakt_core.registry.trainer_registry import register_trainer

@register_trainer("dino")
class DINOTrainer(BaseTrainer):
    def __init__(
        self,
        model,
        train_loader,
        val_loader=None,
        loss_fn=None,
        optimizer_cls=None,
        optimizer_args=None,
        scheduler=None,
        device="cuda",
        **kwargs
    ):
        super().__init__(model, train_loader, val_loader, device)

        if loss_fn is None:
            raise ValueError("loss_fn is required for DINOTrainer")
        self.loss_fn = loss_fn

        if optimizer_cls is None:
            optimizer_cls = torch.optim.Adam
        if optimizer_args is None:
            optimizer_args = {"lr": 1e-3}

        self.optimizer = optimizer_cls(self.model.parameters(), **optimizer_args)
        self.scheduler = scheduler
        self.scaler = GradScaler(enabled=(self.device.type == 'cuda'))

    def _unpack_views(self, batch):
        # Handle default collate format for contrastive learning
        if isinstance(batch, (tuple, list)) and len(batch) == 2:
            if all(isinstance(b, torch.Tensor) for b in batch):
                # Default collate format: (view1_batch, view2_batch)
                return [batch[0].to(self.device).float(),
                        batch[1].to(self.device).float()]
        
        # Original handling for other formats
        if isinstance(batch, torch.Tensor):
            if batch.ndim == 5 and batch.size(1) == 2:
                return [batch[:, 0].to(self.device).float(),
                        batch[:, 1].to(self.device).float()]
            else:
                raise ValueError(f"Unexpected tensor batch shape: {batch.shape}")
        elif isinstance(batch, dict):
            return [
                batch["view1"].to(self.device).float(),
                batch["view2"].to(self.device).float()
            ]
        elif isinstance(batch, (list, tuple)):
            view1_batch = []
            view2_batch = []
            
            for item in batch:
                if isinstance(item, (tuple, list)):
                    view1_batch.append(item[0])
                    view2_batch.append(item[1])
                elif isinstance(item, dict):
                    view1_batch.append(item["view1"])
                    view2_batch.append(item["view2"])
                else:
                    raise TypeError(f"Unexpected batch item type: {type(item)}")
                    
            return [
                torch.stack(view1_batch).to(self.device).float(),
                torch.stack(view2_batch).to(self.device).float()
            ]
        else:
            raise ValueError(f"Unexpected batch type: {type(batch)}")


    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

            for batch in loop:
                try:
                    views = self._unpack_views(batch)

                    with autocast(device_type=self.device.type):
                        student_out = torch.stack(
                            [self.model(view, teacher=False) for view in views], dim=1
                        )
                        teacher_out = self.model(views[0], teacher=True).unsqueeze(1)
                        loss = self.loss_fn(student_out, teacher_out)

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.model.update_teacher()
                    total_loss += loss.item()

                    loop.set_postfix(loss=loss.item())
                except Exception as e:
                    loop.write(f"[ERROR] Batch skipped due to error: {e}")
                    continue

            if self.scheduler:
                self.scheduler.step()

            avg_loss = total_loss / len(self.train_loader)
            print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self):
        if self.val_loader is None:
            print("No validation loader provided")
            return None

        self.model.eval()
        total_loss = 0.0
        loop = tqdm(self.val_loader, desc="Evaluating", leave=True)

        with torch.no_grad():
            for batch in loop:
                try:
                    views = self._unpack_views(batch)
                    student_out = torch.stack(
                        [self.model(view, teacher=False) for view in views], dim=1
                    )
                    teacher_out = self.model(views[0], teacher=True).unsqueeze(1)
                    loss = self.loss_fn(student_out, teacher_out)
                    total_loss += loss.item()

                    loop.set_postfix(val_loss=loss.item())
                except Exception as e:
                    loop.write(f"[ERROR] Validation batch skipped due to error: {e}")
                    continue

        avg_val_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
        return avg_val_loss
