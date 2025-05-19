import os 
import torch
import shutil
import requests
import zipfile
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, List, Dict
from refrakt_core.utils import config
from refrakt_core.utils.config import SCALE_FACTOR
from torchvision import transforms

def patchify(images, n_patches):
    """
    Vectorized implementation of patchify.
    Args:
        images (torch.Tensor): Input images of shape (N, C, H, W).
        n_patches (int): Number of patches along one dimension.
    Returns:
        torch.Tensor: Patches of shape (N, n_patches^2, patch_size^2 * C).
    """
    n, c, h, w = images.shape
    assert h == w, "Only square images supported"
    patch_size = h // n_patches
    unfold = torch.nn.Unfold(kernel_size=patch_size, stride=patch_size)
    patches = unfold(images).transpose(1, 2)
    return patches.reshape(n, n_patches**2, -1)


def positional_embeddings(sequence_length, d):
    """
    Vectorized implementation of positional embeddings.
    Args:
        sequence_length (int): Length of the sequence.
        d (int): Embedding dimension.
    Returns:
        torch.Tensor: Positional embeddings of shape (sequence_length, d).
    """
    position = torch.arange(sequence_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d, 2) * -(np.log(10000.0) / d))
    embeddings = torch.zeros(sequence_length, d)
    embeddings[:, 0::2] = torch.sin(position * div_term)
    embeddings[:, 1::2] = torch.cos(position * div_term)
    return embeddings

def visualize_reconstructions(model, test_loader, type, device, num_samples=5):
    """
    Visualize original and reconstructed images

    Parameters:
    - model: Trained autoencoder model
    - test_loader: DataLoader for test data
    - type: Type of autoencoder
    - device: Computing device
    - num_samples: Number of samples to visualize
    """
    model.eval()
    with torch.no_grad():
        # Get first batch
        data, _ = next(iter(test_loader))
        data = data.to(device).view(data.size(0), -1)

        # Reconstruct images based on autoencoder type
        if type == 'simple':
            reconstructed = model(data)
        elif type == 'regularized':
            reconstructed = model(data)
        elif type == 'denoising':
            noisy_data = data * (1 - config.NOISE_FACTOR) + \
                torch.rand(data.size()).to(device) * config.NOISE_FACTOR
            reconstructed = model(noisy_data)
        elif type == 'vae':
            _, reconstructed, _, _ = model(data)

        # Prepare for visualization
        data = data.cpu().view(-1, 28, 28)
        reconstructed = reconstructed.cpu().view(-1, 28, 28)

        # Plot
        plt.figure(figsize=(10, 4))
        for i in range(num_samples):
            # Original images
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(data[i], cmap='gray')
            plt.title('Original')
            plt.axis('off')

            # Reconstructed images
            plt.subplot(2, num_samples, num_samples + i + 1)
            plt.imshow(reconstructed[i], cmap='gray')
            plt.title('Reconstructed')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'{type}_autoencoder_reconstructions.png')
        plt.close()

## Check these lines

root_dir: Path = Path("/Users/Aksha/github/Re-Implementation/srgan")
dataset_dir: Path = root_dir / "dataset"

def download_dataset(root_dir: Path) -> None:
    if root_dir.exists():
        print("Directory already exists.")
    else:
        print("Creating the directory...")
        root_dir.mkdir(parents=True, exist_ok=True)

        with open(root_dir / "dataset.zip", "wb") as file:
            request = requests.get(
                "https://figshare.com/ndownloader/files/38256855")
            print("Downloading...")
            file.write(request.content)

        with zipfile.ZipFile(root_dir / "dataset.zip", "r") as zip_ref:
            print("Unzipping...")
            zip_ref.extract(root_dir)


def create_dirs():
    dataset_dir = (
        Path.mkdir(root_dir / "dataset") if not Path.exists(dataset_dir) else None
    )
    lr_dir = (
        Path.mkdir(dataset_dir / "lr") if not Path.exists(dataset_dir / "lr") else None
    )
    hr_dir = (
        Path.mkdir(dataset_dir / "hr") if not Path.exists(dataset_dir / "hr") else None
    )


def count_folders(root_dir: Path) -> None:
    for folder in root_dir.iterdir():
        if folder.is_dir():
            folder_count, file_count = count_folders(folder)

            print(f"Directory: {folder}")
            print(f" Number of folders: {folder_count+1}")
            print(f"Number of files: {file_count+1}")
        else:
            print(f"File found {folder}")
    return 0, 0


def move_images(lr_dir: Path, hr_dir: Path, root_dir: Path):
    for folder in root_dir.iterdir():
        if folder.is_dir():
            for file in folder.iterdir():
                if file.name.endswith("_LR.png"):
                    source = file
                    destination = lr_dir / file.name
                    source.rename(destination)
                elif file.name.endswith("_HR.png"):
                    source = file
                    destination = hr_dir / file.name
                    source.rename(destination)


def delete_dir(dir: Path):
    try:
        shutil.rmtree(dir)
        print(f"Removed directory: {dir}")
    except OSError as e:
        print(f"Error: {e.strerror}")


def get_transform(is_hr=True, scale_factor=SCALE_FACTOR):
    if is_hr:
        return transforms.Compose([
            transforms.Resize((32 * scale_factor, 32 * scale_factor)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    This function gives the result of the classes present inside the custom dataset.

    Argument:
    directory: The directory of the dataset passed.

    Returns a two-length tuple of List type and Dict type.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't load any classes in {directory}")

    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx