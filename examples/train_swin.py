import os
import sys
import torch
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from trainer import Trainer
import models

def main():
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=transform_train
    )
    
    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=transform_val
    )
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=64, num_workers=4)

    trainer = Trainer(
        model_name="swin",
        model_args={"num_classes": 10},
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=nn.CrossEntropyLoss(),
        optimizer=optim.AdamW,
        device="cpu"
    )

    
    trainer.optimizer = trainer.optimizer(
        trainer.model.parameters(), 
        lr=5e-4,
        weight_decay=0.05,
        betas=(0.9, 0.999)
    )
    
    trainer.train(num_epochs=30)
    trainer.evaluate()

if __name__ == "__main__":
    main()