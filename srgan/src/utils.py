import shutil
import requests
import zipfile
from pathlib import Path
from torchvision import transforms
from config import SCALE_FACTOR

root_dir = Path("/Users/Aksha/github/Re-Implementation/srgan")
dataset_dir = root_dir / "dataset"


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
        Path.mkdir(dataset_dir /
                   "lr") if not Path.exists(dataset_dir / "lr") else None
    )
    hr_dir = (
        Path.mkdir(dataset_dir /
                   "hr") if not Path.exists(dataset_dir / "hr") else None
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
