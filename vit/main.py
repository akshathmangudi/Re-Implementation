from pathlib import Path
import torch
import numpy as np
from torch.nn import CrossEntropyLoss
from torch.utils.data import DataLoader
# Uncomment this line for MNIST training.
# from torchvision.datasets.mnist import MNIST
from torchvision import transforms
from tqdm import tqdm, trange
from torch.optim import Adam

from utils.data_setup import data_setup
from utils.dataloader import CreateDataset
from models.vit import ViT

np.random.seed(42)
torch.manual_seed(42)


def main(train_loader, test_loader):
    """
    This code contains the training and testing loop for training the vision transformers model. It requires two
    parameters

    :param train_loader: The dataloader for the training set for training the model.
    :param test_loader: The dataloader for the testing set during evaluation phase.
    """
    # For MNIST, comment out "food_model" and uncomment "mnist_model" and replace any instances of such.

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    food_model = ViT((3, 64, 64), n_patches=8, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    # mnist_model = ViT((1, 28, 28), n_patches=7, n_blocks=2, hidden_d=8, n_heads=2, out_d=10).to(device)
    print("Using device: ", device, f"({torch.cuda.get_device_name(device)})" if torch.cuda.is_available() else "")

    epochs = 8
    lr = 0.005

    optimizer = Adam(food_model.parameters(), lr=lr)
    criterion = CrossEntropyLoss()
    for epoch in trange(epochs, desc="train"):
        train_loss = 0.0
        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1} in training", leave=False):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = food_model(x)
            loss = criterion(y_hat, y)

            train_loss += loss.detach().cpu().item() / len(train_loader)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Epoch {epoch + 1}/{epochs} loss: {train_loss:.2f}")

    with torch.no_grad():
        correct, total = 0, 0
        test_loss = 0.0
        for batch in tqdm(test_loader, desc="Testing"):
            x, y = batch
            x, y = x.to(device), y.to(device)
            y_hat = food_model(x)
            loss = criterion(y_hat, y)
            test_loss += loss.detach().cpu().item() / len(test_loader)

            correct += torch.sum(torch.argmax(y_hat, dim=1) == y).detach().cpu().item()
            total += len(x)

        print(f"Test loss: {test_loss:.2f}")
        print(f"Test accuracy: {correct / total * 100:.2f}%")


if __name__ == "__main__":
    """
        I have used a custom dataset for this section. 
        Details on how to run this on MNIST is listed below. It is arguably simpler and yields more accuracy.
    """
    data_path = Path("dataset/")
    image_path = data_path / "FoodClass"
    data_setup(data_path, image_path)

    train_dir = image_path / "train"
    test_dir = image_path / "test"

    # Defining the transformation for the custom dataset.
    transformation = transforms.Compose([
        transforms.Resize(size=(64, 64)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ToTensor()
    ])

    train_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(p=0.4),
        transforms.ToTensor()
    ])

    test_transforms = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor()
    ])

    # Using the transformation for our training and testing dataset, as well as creating our DataLoaders.
    train_data = CreateDataset(target_dir=str(train_dir),
                               transform=train_transforms)

    test_data = CreateDataset(target_dir=str(test_dir),
                              transform=test_transforms)

    train_dataloader = DataLoader(dataset=train_data,
                                  batch_size=1,
                                  num_workers=0,
                                  shuffle=True)

    test_dataloader = DataLoader(dataset=test_data,
                                 batch_size=1,
                                 num_workers=0,
                                 shuffle=False)

    img_custom, label_custom = next(iter(train_dataloader))

    # Running the testing and training loop.
    main(train_loader=train_dataloader, test_loader=test_dataloader)

    # For MNIST: comment out the lines above and uncomment the lines below.

    # transform = ToTensor()
    # train_mnist = MNIST(root='./mnist', train=True, download=True, transform=transform)
    # test_mnist = MNIST(root='./mnist', train=False, download=True, transform=transform)
    # train_loader = DataLoader(train_mnist, shuffle=True, batch_size=128)
    # test_loader = DataLoader(test_mnist, shuffle=False, batch_size=128)
    # main(train_loader=train_loader, test_loader=test_loader)
