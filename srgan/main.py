from pathlib import Path
from src.utils import download_dataset, create_dirs, count_folders, delete_dir, move_images

if __name__ == "__main__":
    # Creating our basic structure
    root_dir = Path.cwd()
    dataset_dir = root_dir / "dataset"

    # Calling our download dataset and create lr, hr directories.
    download_dataset(root_dir=root_dir)
    create_dirs()

    # Initializing them for the next few function calls. 
    set14_dir = root_dir / "Set14"
    hr_dir = dataset_dir / "hr"
    lr_dir = dataset_dir / "lr"

    # Moving from Set14/ to dataset/hr/ and dataset/lr/
    count_folders(root_dir)
    move_images(lr_dir=lr_dir, hr_dir=hr_dir, root_dir=root_dir)

    # Deleting the non-empty Set14/ directory.
    delete_dir(set14_dir)