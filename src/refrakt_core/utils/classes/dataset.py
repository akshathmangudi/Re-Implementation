import torch
from PIL import Image
from pathlib import Path
from typing import Tuple
from torch.utils.data import Dataset
from refrakt_core.utils.methods import find_classes


class CreateDataset(Dataset):
    """
    This dataset class was created as described by the documentation 
    provided by PyTorch. Most of the details here are explanatory. 
    """
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


class FlattenTransform:
    def __call__(self, x):
        return torch.flatten(x)
    