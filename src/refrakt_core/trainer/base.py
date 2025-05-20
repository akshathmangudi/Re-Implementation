from abc import ABC, abstractmethod
import torch

class BaseTrainer(ABC):
    def __init__(self, model, train_loader, val_loader, device="cuda"):
        self.device = torch.device(device)
        self.model = model.to(self.device)
        self.train_loader = train_loader
        self.val_loader = val_loader

    @abstractmethod
    def train(self, num_epochs):
        pass

    @abstractmethod
    def evaluate(self):
        pass
