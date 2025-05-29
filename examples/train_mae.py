import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root / "src"))

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from refrakt_core.trainer.autoencoder import AETrainer
from refrakt_core.registry.model_registry import get_model
from refrakt_core.losses.mae import MAELoss
from refrakt_core.transforms import PatchifyTransform

import refrakt_core.models
import refrakt_core.trainer
import refrakt_core.losses

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # === Parameters ===
    image_size = 224
    patch_size = 16
    batch_size = 64
    num_epochs = 100
    mask_ratio = 0.75

    # === Transform: Resize, Normalize, Patchify ===
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
        PatchifyTransform(patch_size=patch_size)  # Custom transform if needed
    ])

    # === Dataset: Use CIFAR10 or similar small dataset for demo ===
    from torchvision.datasets import CIFAR10
    train_dataset = CIFAR10(root="./data", train=True, download=True, transform=transform)
    test_dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # === Model ===
    model = get_model(
        "mae",
        image_size=image_size,
        patch_size=patch_size,
        in_chans=3,
        embed_dim=768,
        encoder_depth=12,
        decoder_dim=512,
        decoder_depth=8,
        num_heads=12,
        decoder_num_heads=16,
        mask_ratio=mask_ratio
    ).to(device)

    # === Loss ===
    loss_fn = MAELoss(normalize_target=False)

    # === Trainer ===
    trainer = AETrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer_cls=torch.optim.AdamW,
        optimizer_args={"lr": 1.5e-4, "betas": (0.9, 0.95), "weight_decay": 0.05},
        device=device
    )

    trainer.train(num_epochs=num_epochs)
    trainer.evaluate()

if __name__ == "__main__":
    main()
