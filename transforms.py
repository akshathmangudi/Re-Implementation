import random
import torchvision.transforms as T


class PairedTransform:
    # For super-resolution datasets
    def __init__(self, crop_size=96):
        self.crop_size = crop_size

    def __call__(self, lr, hr):
        # Random crop
        i, j, h, w = T.RandomCrop.get_params(hr, output_size=(self.crop_size * 4, self.crop_size * 4))
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
