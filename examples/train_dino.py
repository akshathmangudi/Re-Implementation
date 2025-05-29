import os
import sys
import torch
from pathlib import Path

# Ensure access to refrakt_core
project_root = Path(__file__).parent.parent.resolve()
sys.path.append(str(project_root / "src"))

from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

from refrakt_core.trainer.dino import DINOTrainer
from refrakt_core.datasets import ContrastiveDataset
from refrakt_core.losses.dino import DINOLoss
from refrakt_core.registry.model_registry import get_model
import refrakt_core.models  # triggers model registration

def debug_batch(batch, device):
    """Debug function to check tensor devices in batch"""
    print(f"Batch type: {type(batch)}")
    if isinstance(batch, (list, tuple)):
        print(f"Number of items in batch: {len(batch)}")
        for i, item in enumerate(batch):
            if isinstance(item, torch.Tensor):
                print(f"  Item {i}: {item.shape}, device: {item.device}, dtype: {item.dtype}")
            else:
                print(f"  Item {i}: {type(item)}")
    elif isinstance(batch, torch.Tensor):
        print(f"Single tensor: {batch.shape}, device: {batch.device}, dtype: {batch.dtype}")
    else:
        print(f"Unknown batch type: {type(batch)}")

def main():
    # Define standard DINO augmentations
    dino_transform = transforms.Compose([
        transforms.RandomResizedCrop(32),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ToTensor(),  # Convert to tensor before ColorJitter
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR-10 dataset
    train_dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=None)
    test_dataset = datasets.CIFAR10(root="./data", train=False, download=True, transform=None)

    # Wrap with ContrastiveDataset (returns two views per sample)
    contrastive_train_dataset = ContrastiveDataset(train_dataset, transform=dino_transform)
    contrastive_test_dataset = ContrastiveDataset(test_dataset, transform=dino_transform)

    train_loader = DataLoader(contrastive_train_dataset, batch_size=128, shuffle=True, drop_last=True, num_workers=4)
    val_loader = DataLoader(contrastive_test_dataset, batch_size=128, shuffle=False, num_workers=4)

    # Debug: Check a single batch
    print("=== DEBUGGING BATCH ===")
    for batch in train_loader:
        debug_batch(batch, "cuda")
        break

    # Instantiate the DINO model
    try:
        model = get_model("dino", backbone="resnet18", out_dim=65536)
    except Exception as e:
        print(f"[ERROR] Failed to initialize model: {e}")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    # Move model to device
    model = model.to(device)
    print(f"Model device: {next(model.parameters()).device}")

    # Instantiate DINO-specific loss
    loss_fn = DINOLoss(out_dim=65536, teacher_temp=0.04, student_temp=0.1, center_momentum=0.9)
    loss_fn = loss_fn.to(device)
    print(f"Loss center device: {loss_fn.center.device}")

    # Debug: Test a single forward pass
    print("=== TESTING FORWARD PASS ===")
    model.eval()
    with torch.no_grad():
        for batch in train_loader:
            try:
                print("Raw batch:")
                debug_batch(batch, device)
                
                # Try to move to device
                if isinstance(batch, (list, tuple)):
                    views = [v.to(device) if isinstance(v, torch.Tensor) else v for v in batch]
                else:
                    views = [batch.to(device)]
                
                print("After moving to device:")
                for i, view in enumerate(views):
                    if isinstance(view, torch.Tensor):
                        print(f"  View {i}: {view.shape}, device: {view.device}")
                
                # Test forward pass
                if len(views) > 0 and isinstance(views[0], torch.Tensor):
                    print("Testing model forward pass...")
                    student_out = model(views[0], teacher=False)
                    teacher_out = model(views[0], teacher=True)
                    print(f"Student out: {student_out.shape}, device: {student_out.device}")
                    print(f"Teacher out: {teacher_out.shape}, device: {teacher_out.device}")
                    print("Forward pass successful!")
                
                break
            except Exception as e:
                print(f"Error in forward pass: {e}")
                import traceback
                traceback.print_exc()
                break

    # Optimizer and optional scheduler
    optimizer = optim.Adam(model.parameters(), lr=3e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)

    # DINO trainer
    trainer = DINOTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        loss_fn=loss_fn,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    # Train + evaluate
    try:
        print("=== STARTING TRAINING ===")
        trainer.train(num_epochs=1)
        trainer.evaluate()
    except Exception as e:
        print(f"[ERROR] Training or evaluation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()  