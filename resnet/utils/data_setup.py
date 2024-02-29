from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10


def data_loader(data_dir,
                batch_size,
                test=False):
    """
    This function creates the dataloader for our dataset. In this implementation, I am using the CIFAR10 dataset for
    image classification.

    Arguments:
    data_dir: The directory of our dataset.
    batch_size: Our batch size to split up the dataset.
    test: If the dataset is a testing dataset, we will not shuffle. Else, we will.
    """

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        normalize,
    ])

    if test:
        dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        return dataloader

    train_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    valid_dataset = CIFAR10(root=data_dir, train=True, download=True, transform=transform)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader, valid_dataloader
