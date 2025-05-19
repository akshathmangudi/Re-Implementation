import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from trainer import Trainer
from registry.model_registry import get_model
from utils.classes import FlattenTransform

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        FlattenTransform()
    ])
    
    # MNIST dataset
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model_args = dict(
        input_dim=784,
        hidden_dim=32,
        type="simple",
    )
    
    trainer = Trainer(
        model_name="autoencoder",
        model_args=model_args,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.MSELoss(),
        optimizer=optim.Adam,
        device=device
    )
    
    trainer.optimizer = trainer.optimizer(trainer.model.parameters(), lr=1e-3)
    
    trainer.train(num_epochs=1)
    trainer.evaluate()

if __name__ == "__main__":
    main()