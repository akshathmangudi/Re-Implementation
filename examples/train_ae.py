import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root / "src"))


import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from refrakt_core.trainer.autoencoder import AETrainer
from refrakt_core.registry.model_registry import get_model
from refrakt_core.transforms import FlattenTransform

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
        FlattenTransform()
    ])
    
    train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    model = get_model("autoencoder", input_dim=784, hidden_dim=32, type="vae")

    trainer = AETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.MSELoss(),  # or custom loss for KL + recon
        optimizer_cls=optim.Adam,
        optimizer_args={"lr": 1e-3},
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer.train(num_epochs=10)
    trainer.evaluate()

if __name__ == "__main__":
    main()