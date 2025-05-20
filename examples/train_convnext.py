import os
import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root / "src"))

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from refrakt_core.trainer.supervised import SupervisedTrainer
from refrakt_core.registry.model_registry import get_model
import refrakt_core.models 

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

    model = get_model(
        "convnext",
        in_channels=1,
        num_classes=10
    )

    trainer = SupervisedTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer_cls=optim.AdamW,
        optimizer_args={"lr": 3e-4},
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer.train(num_epochs=1)
    trainer.evaluate()

if __name__ == "__main__":
    main()
