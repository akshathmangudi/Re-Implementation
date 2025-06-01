# train_msn.py

import os
import sys
from pathlib import Path

import torch

project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root / "src"))

from torch import optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

import refrakt_core.losses
import refrakt_core.models  # Needed to trigger @register_model decorators
import refrakt_core.trainer
from refrakt_core.losses.msn import MSNLoss
from refrakt_core.registry.model_registry import get_model
from refrakt_core.trainer.msn import MSNTrainer


def main():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    dataset = datasets.FakeData(size=512, image_size=(3, 224, 224), transform=transform)
    train_loader = DataLoader(dataset, batch_size=128, shuffle=True)

    model = get_model(
        "msn",
        encoder_name="vit_base_patch16_224",
        projector_dim=256,
        num_prototypes=1024,
        pretrained=False
    )

    trainer = MSNTrainer(
        model=model,
        train_loader=train_loader,
        loss_fn=MSNLoss(temp_anchor=0.1, temp_target=0.04, lambda_me_max=1.0),
        optimizer_cls=optim.AdamW,
        optimizer_args={"lr": 3e-4, "weight_decay": 0.05},
        device="cuda" if torch.cuda.is_available() else "cpu"
    )

    trainer.train(num_epochs=10)

if __name__ == "__main__":
    main()
