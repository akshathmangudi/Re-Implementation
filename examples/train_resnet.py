# refrakt/examples/train_resnet.py
import sys
import os
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))
import models

import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from trainer import Trainer

def main():
    # Data transforms
    transform = transforms.Compose([
        transforms.Resize(224),  # ResNet expects larger images
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])
    ])

    # CIFAR-10 example (adjust paths as needed)
    train_dataset = datasets.CIFAR10(
        root='./data', 
        train=True, 
        download=True, 
        transform=transform
    )
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=64)

    # Initialize Trainer for ResNet-18
    trainer = Trainer(
        model_name="resnet18",
        model_args={
            "in_channels": 3,
            "num_classes": 10
        },
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optim.Adam,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer.optimizer = trainer.optimizer(trainer.model.parameters(), lr=1e-3)

    trainer.train(num_epochs=10)
    trainer.evaluate()

if __name__ == "__main__":
    main()