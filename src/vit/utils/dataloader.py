import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset
from typing import Tuple, List, Dict


def find_classes(directory: str) -> Tuple[List[str], Dict[str, int]]:
    """
    This function gives the result of the classes present inside the custom dataset. It has one parameter.

    :param directory: The directory of the dataset passed.
    :return: Returns a two-length tuple of List type and Dict type.
    """
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())

    if not classes:
        raise FileNotFoundError(f"Couldn't load any classes in {directory}")

    class_to_idx = {class_name: i for i, class_name in enumerate(classes)}
    return classes, class_to_idx


class CreateDataset(Dataset):
    def __init__(self, target_dir: str, transform=None) -> None:
        self.paths = list(Path(target_dir).glob("*/*.jpg"))
        self.transform = transform
        self.classes, self.class_to_idx = find_classes(target_dir)

    def load_image(self, index: int) -> Image.Image:
        image_path = self.paths[index]
        return Image.open(image_path)

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        image = self.load_image(index)
        class_name = self.paths[index].parent.name
        class_idx = self.class_to_idx[class_name]

        if self.transform:
            return self.transform(image), class_idx
        else:
            return image, class_idx