# refrakt_examples/mnist_vit.py
import os 
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.resolve()))
from refrakt import models

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from refrakt.trainer import Trainer

# 1. MNIST Data Loading
transform = transforms.Compose([
    transforms.Resize(28),  # ViT needs square images
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST stats
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
test_loader = DataLoader(test_dataset, batch_size=128)

# 2. Configure ViT for MNIST
trainer = Trainer(
    model_name="vit",
    train_loader=train_loader,
    val_loader=test_loader,
    loss_fn=nn.CrossEntropyLoss(),
    optimizer=optim.Adam,
    model_args={
        "image_size": 28,    
        "patch_size": 4,     
        "num_classes": 10,   
        "dim": 64,           
        "depth": 6,          
        "heads": 4,          
        "in_channels": 1
    },
    device = "cuda"
)

# 3. Train and evaluate
trainer.train(num_epochs=10)
trainer.evaluate()