import random

import torch
import torchvision.transforms as T
from torchvision.transforms import functional as TF

from refrakt_core.registry.transform_registry import register_transform


@register_transform("paired")
class PairedTransform:
    # For super-resolution datasets
    def __init__(self, crop_size=96):
        self.crop_size = crop_size

    def __call__(self, lr, hr):
        # Random crop
        i, j, h, w = T.RandomCrop.get_params(
            hr, output_size=(self.crop_size * 4, self.crop_size * 4)
        )
        hr = T.functional.crop(hr, i, j, h, w)
        lr = T.functional.crop(lr, i // 4, j // 4, h // 4, w // 4)

        # Random horizontal flip
        if random.random() > 0.5:
            hr = T.functional.hflip(hr)
            lr = T.functional.hflip(lr)

        # ToTensor
        hr = T.ToTensor()(hr)
        lr = T.ToTensor()(lr)

        return lr, hr


@register_transform("flatten")
class FlattenTransform:
    def __call__(self, x):
        return torch.flatten(x)


@register_transform("patchify")
class PatchifyTransform:
    def __init__(self, patch_size):
        self.patch_size = patch_size

    def __call__(self, img):
        # img: Tensor shape [C, H, W]
        c, h, w = img.shape
        p = self.patch_size
        assert h % p == 0 and w % p == 0, "Image dims must be divisible by patch size"
        return img  # patchify is handled in model internally
