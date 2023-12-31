import requests
import zipfile
from pathlib import Path


def data_setup(
        data_path: Path,
        image_path: Path
):
    # If data path doesn't exist, download and create one...
    """
    This function performs the download part of the Food101 dataset. If the dataset folder already exists, no issue. Else
    the function uses the request library to download the zip file and extracts it.

    Arguments:
    data_path: The root path 'dataset'
    image_path: The dataset path 'Food101'

    These paths are checked for their existence and if it fails, a new folder is created.
    """
    if data_path.is_dir():
        print(f"{data_path}\ exists already.")
    else:
        print(f"{data_path} does not exist, creating one.")
        data_path.mkdir(parents=True, exist_ok=True)

    # Same goes for image_path
    if image_path.is_dir():
        print(f"{image_path} already exists.")
    else:
        print(f"{image_path} not found. Creating one...")
        image_path.mkdir(parents=True, exist_ok=True)

        # Using requests to download from link.
        with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
            request = requests.get(
                "https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
            print("Downloading pizza, steak, sushi data...")
            f.write(request.content)

        with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_file:
            print("Unzipping...")
            zip_file.extractall(image_path)