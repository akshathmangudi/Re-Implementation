import os 
from PIL import Image
from pathlib import Path
import torchvision.transforms as T
from torch.utils.data import Dataset

class ContrastiveDataset(Dataset):
    def __init__(self, base_dataset, transform):
        self.base_dataset = base_dataset
        self.transform = transform
        
    def __getitem__(self, index):
        x, _ = self.base_dataset[index]
        return self.transform(x), self.transform(x)
        
    def __len__(self):
        return len(self.base_dataset)


class SuperResolutionDataset(Dataset):
    def __init__(self, lr_dir, hr_dir, transform=None):
        self.lr_dir = Path(lr_dir)
        self.hr_dir = Path(hr_dir)
        self.filenames = sorted(os.listdir(self.lr_dir))
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        lr = Image.open(self.lr_dir / fname).convert("RGB")
        hr = Image.open(self.hr_dir / fname).convert("RGB")
        
        if self.transform:
            lr, hr = self.transform(lr, hr)
        
        return {"lr": lr, "hr": hr}