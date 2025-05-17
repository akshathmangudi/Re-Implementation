import torch
import torch.nn as nn
import gc

from models.resnet import ResNet, ResidualBlock
from utils.data_setup import data_loader


def main(train_dataloader, valid_dataloader, test_dataloader, num_classes, num_epochs, batch_size, learning_rate,
         device, model):
    """
    This function is responsible for the train, validation and test loop for our model. This contains several
    parameters.

    Our total dataset has been split into 80% training, 10% validation and 10% testing.

    Arguments:
    train_dataloader: The dataloader for our training set.
    valid_dataloader: The dataloader for our validation set.
    test_dataloader: The dataloader for our testing set.
    num_classes: The number of classes in the dataset. For CIFAR10, there are 10 different classes.
    num_epochs: The number of passes we'll choose for our dataset.
    batch_size: The number that we'll be splitting the dataset into.
    learning_rate: Our learning rate for our loss function
    device: Whether to run on the GPU or CPU.
    model: Our ResNet model.
    """

    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_dataloader):
            # Move tensors to the configured device
            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            del images, labels, outputs
            torch.cuda.empty_cache()
            gc.collect()

        print('Epoch [{}/{}], Loss: {:.4f}'
              .format(epoch + 1, num_epochs, loss.item()))

        # Validation
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in valid_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} validation images: {} %'.format(5000, 100 * correct / total))

            # Testing
        with torch.no_grad():
            correct = 0
            total = 0
            for images, labels in test_dataloader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                del images, labels, outputs

            print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))


if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_loader, valid_loader = data_loader(data_dir='./cifar10', batch_size=64)
    test_loader = data_loader(data_dir='./cifar10', batch_size=64, test=True)
    model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=0.001, momentum=0.9)

    # Train the model
    total_step = len(train_loader)
    main(train_dataloader=train_loader, valid_dataloader=valid_loader, test_dataloader=test_loader, num_classes=10,
         num_epochs=20, batch_size=16, learning_rate=0.01, device=device, model=model)
