import torch
from torch.cuda.amp import GradScaler, autocast
from refrakt_core.trainer.base import BaseTrainer
import torch.nn.functional as F
from tqdm import tqdm  # ✅ Add tqdm for progress bars


class DINOTrainer(BaseTrainer):
    """
    Trainer for DINO framework with mixed precision and teacher momentum updates.
    """

    def __init__(self, model, train_loader, val_loader, loss_fn, optimizer, scheduler=None, device="cuda"):
        super().__init__(model, train_loader, val_loader, device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.scaler = GradScaler()

    def train(self, num_epochs):
        for epoch in range(num_epochs):
            self.model.train()
            total_loss = 0.0

            # ✅ Add tqdm progress bar for training
            loop = tqdm(self.train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=True)

            for batch in loop:
                try:
                    views = [v.to(self.device) for v in batch]  # list of multi-views
                    with autocast():
                        student_out = torch.stack(
                            [self.model(view, teacher=False) for view in views], dim=1
                        )  # (B, num_views, out_dim)
                        teacher_out = self.model(views[0], teacher=True).unsqueeze(1)  # (B, 1, out_dim)
                        loss = self.loss_fn(student_out, teacher_out)

                    self.optimizer.zero_grad()
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                    self.model.update_teacher()
                    total_loss += loss.item()

                    # ✅ Update tqdm with current loss
                    loop.set_postfix(loss=loss.item())
                except Exception as e:
                    loop.write(f"[ERROR] Batch skipped due to error: {e}")
                    continue

            if self.scheduler:
                self.scheduler.step()

            avg_loss = total_loss / len(self.train_loader)
            print(f"\nEpoch [{epoch+1}/{num_epochs}], Loss: {avg_loss:.4f}")

    def evaluate(self):
        self.model.eval()
        total_loss = 0.0

        # ✅ Add tqdm progress bar for validation
        loop = tqdm(self.val_loader, desc="Evaluating", leave=True)

        with torch.no_grad():
            for batch in loop:
                try:
                    views = [v.to(self.device) for v in batch]
                    student_out = torch.stack(
                        [self.model(view, teacher=False) for view in views], dim=1
                    )
                    teacher_out = self.model(views[0], teacher=True).unsqueeze(1)
                    loss = self.loss_fn(student_out, teacher_out)
                    total_loss += loss.item()

                    # ✅ Update tqdm with current loss
                    loop.set_postfix(val_loss=loss.item())
                except Exception as e:
                    loop.write(f"[ERROR] Validation batch skipped due to error: {e}")
                    continue

        avg_val_loss = total_loss / len(self.val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}")
