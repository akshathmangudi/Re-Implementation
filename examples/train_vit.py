# examples/train_vit.py

import os
import sys
import torch
from pathlib import Path

# Add project root to sys.path for clean imports
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from trainer import Trainer
import refrakt.models

def main():
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST(
        root='./data',
        train=True,
        download=True,
        transform=transform
    )

    test_dataset = datasets.MNIST(
        root='./data',
        train=False,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=128)

    trainer = Trainer(
        model_name="vit",
        model_args=dict(
            image_size=28,
            patch_size=4,
            num_classes=10,
            dim=64,
            depth=6,
            heads=4,
            in_channels=1,
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optim.AdamW,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer.optimizer = trainer.optimizer(trainer.model.parameters(), lr=1e-3)

    trainer.train(num_epochs=1)
    trainer.evaluate()

if __name__ == "__main__":
    main()
