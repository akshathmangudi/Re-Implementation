# examples/train_simclr.py

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
from torchvision.transforms import functional as TF
from torch.utils.data import Dataset

from trainer import Trainer
import models
from utils.classes import ContrastiveDataset
from losses.ntxent import NTXentLoss

def main():
    # Define strong augmentations for contrastive learning
    contrastive_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    
    base_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=None
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=None
    )
    
    # Wrap datasets for contrastive learning
    contrastive_train_dataset = ContrastiveDataset(train_dataset, contrastive_transform)
    contrastive_test_dataset = ContrastiveDataset(test_dataset, contrastive_transform)

    train_loader = DataLoader(contrastive_train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(contrastive_test_dataset, batch_size=128)

    trainer = Trainer(
        model_name="simclr",
        model_args=dict(
            proj_dim=128
        ),
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=NTXentLoss(temperature=0.5),
        optimizer=optim.Adam,
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer.optimizer = trainer.optimizer(trainer.model.parameters(), lr=3e-4)

    trainer.train(num_epochs=1)
    trainer.evaluate()

if __name__ == "__main__":
    main()