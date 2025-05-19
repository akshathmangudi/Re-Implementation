import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root))

import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from trainer import Trainer
from datasets import SuperResolutionDataset
from transforms import PairedTransform
from losses.gan import GANLoss
from registry.model_registry import get_model
import models

def get_srgan_dataloaders(data_root, batch_size=16, num_workers=4, crop_size=96):
    transform = PairedTransform(crop_size)
    
    train_ds = SuperResolutionDataset(
        lr_dir=f"{data_root}/train/LR",
        hr_dir=f"{data_root}/train/HR",
        transform=transform
    )
    val_ds = SuperResolutionDataset(
        lr_dir=f"{data_root}/val/LR",
        hr_dir=f"{data_root}/val/HR",
        transform=transform
    )
    
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader = get_srgan_dataloaders(data_root="./data/DIV2K")
    
    model_args = dict(
        scale_factor=4,
    )
    model = get_model("srgan", **model_args).to(device)
    trainer = Trainer(
        model_name="srgan",
        model_args={"scale_factor": 4},
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn={
            "generator": nn.L1Loss(),
            "discriminator": GANLoss()
        },
        optimizer={
            "generator": torch.optim.Adam(model.generator.parameters(), lr=1e-4),
            "discriminator": torch.optim.Adam(model.discriminator.parameters(), lr=1e-4)
        },
        device=device
    )
    trainer.train(num_epochs=1)
    trainer.evaluate()

if __name__ == "__main__":
    main()
