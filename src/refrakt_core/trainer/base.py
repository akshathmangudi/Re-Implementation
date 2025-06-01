import os
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

import torch


class BaseTrainer(ABC):
    def __init__(self, model, train_loader, val_loader, device="cuda", **kwargs):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.save_dir = kwargs.pop('save_dir', 'checkpoints/')
        self.model_name = kwargs.pop('model_name', 'model')
        self.optimizer = None  # Initialize to None
        self.scheduler = None  # Initialize to None

        
    @abstractmethod
    def train(self, num_epochs):
        pass

    @abstractmethod
    def evaluate(self):
        pass

    def get_checkpoint_path(self, suffix="final") -> str:
        if suffix == "best_model":
            return os.path.join(self.save_dir, f"{self.model_name}.pth")
        else:
            return os.path.join(self.save_dir, f"{self.model_name}_{suffix}.pth")
    
    def save(self, path: Optional[str] = None, suffix: str = "final"):
        if path is None:
            path = self.get_checkpoint_path(suffix)
        dir_path = os.path.dirname(path)
        if dir_path:
            os.makedirs(dir_path, exist_ok=True)
        
        try:
            checkpoint: Dict[str, Any] = {
                'model_state_dict': self.model.state_dict(),
                'model_name': self.model_name
            }
            
            # Handle different optimizer types
            if self.optimizer is not None:
                if isinstance(self.optimizer, dict):
                    checkpoint['optimizer_state_dict'] = {
                        k: v.state_dict() for k, v in self.optimizer.items()
                    }
                else:
                    checkpoint['optimizer_state_dict'] = self.optimizer.state_dict()
            
            # Handle scheduler if exists
            if self.scheduler is not None:
                if isinstance(self.scheduler, dict):
                    checkpoint['scheduler_state_dict'] = {
                        k: v.state_dict() for k, v in self.scheduler.items()
                    }
                else:
                    checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
            
            torch.save(checkpoint, path)
            print(f"[INFO] Model saved to: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to save model: {e}")

    def load(self, path: Optional[str] = None, suffix: str = "final"):
        if path is None:
            path = self.get_checkpoint_path(suffix)
        try:
            checkpoint = torch.load(path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            # Load optimizer state if exists and optimizer is set
            if self.optimizer is not None and 'optimizer_state_dict' in checkpoint:
                optimizer_state = checkpoint['optimizer_state_dict']
                if isinstance(self.optimizer, dict):
                    for k in self.optimizer:
                        if k in optimizer_state:
                            self.optimizer[k].load_state_dict(optimizer_state[k])
                else:
                    self.optimizer.load_state_dict(optimizer_state)
            
            # Load scheduler state if exists and scheduler is set
            if self.scheduler is not None and 'scheduler_state_dict' in checkpoint:
                scheduler_state = checkpoint['scheduler_state_dict']
                if isinstance(self.scheduler, dict):
                    for k in self.scheduler:
                        if k in scheduler_state:
                            self.scheduler[k].load_state_dict(scheduler_state[k])
                else:
                    self.scheduler.load_state_dict(scheduler_state)
                    
            print(f"[INFO] Model loaded from: {path}")
        except Exception as e:
            print(f"[ERROR] Failed to load model: {e}")