import os
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root / "src"))

from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import refrakt_core.models
from refrakt_core.datasets import ContrastiveDataset
from refrakt_core.losses.ntxent import NTXentLoss
from refrakt_core.registry.model_registry import get_model
from refrakt_core.trainer.contrastive import ContrastiveTrainer


def main():
    contrastive_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Load CIFAR-10 without transform (it will be handled in ContrastiveDataset)
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

    # Wrap CIFAR-10 in ContrastiveDataset
    contrastive_train_dataset = ContrastiveDataset(train_dataset, contrastive_transform)
    contrastive_test_dataset = ContrastiveDataset(test_dataset, contrastive_transform)

    train_loader = DataLoader(contrastive_train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(contrastive_test_dataset, batch_size=128)

    # Instantiate the model
    model = get_model("simclr", proj_dim=128)

    # Instantiate the trainer
    trainer = ContrastiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=NTXentLoss(temperature=0.5),
        optimizer_cls=optim.Adam,
        optimizer_args={"lr": 3e-4},
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer.train(num_epochs=1)
    trainer.evaluate()

if __name__ == "__main__":
    main()
