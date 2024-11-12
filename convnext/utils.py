from torch.utils.data import DataLoader


def create_dataloader(transform, train_dataset, test_dataset):
    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    return train_loader, test_loader
