import torch
import config
import torch.nn as nn
import torch.optim as optim
from model import AutoEncoder
from utils import visualize_reconstructions
from torchvision import transforms, datasets


def train(model, train_loader, test_loader=None, type='simple',
          device=config.DEVICE,
          n_epochs=config.NUM_EPOCHS,
          learning_rate=config.LEARNING_RATE,
          weight_decay=config.WEIGHT_DECAY):
    """
    Generic training function for different types of autoencoders
    """
    model = model.to(device)

    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    reconstruction_loss = nn.MSELoss(reduction='sum')

    for epoch in range(n_epochs):
        model.train()
        total_loss = 0.0

        for _, (data, _) in enumerate(train_loader):
            data = data.to(device).view(data.size(0), -1)
            optimizer.zero_grad()

            if type == 'simple':
                reconstructed = model(data)
                loss = reconstruction_loss(reconstructed, data)

            elif type == 'regularized':
                reconstructed = model(data)
                loss = reconstruction_loss(reconstructed, data)

            elif type == 'denoising':
                noisy_data = data * (1 - config.NOISE_FACTOR) + \
                    torch.rand(data.size()).to(device) * config.NOISE_FACTOR
                reconstructed = model(noisy_data)
                loss = reconstruction_loss(reconstructed, data)

            elif type == 'vae':
                encoded, reconstructed, mu, sigma = model(data)
                recon_loss = reconstruction_loss(reconstructed, data)
                kld_loss = -0.5 * \
                    torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
                loss = recon_loss + kld_loss

            else:
                raise ValueError(f"Unknown autoencoder type: {type}")

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader.dataset)
        print(f"Epoch [{epoch+1}/{n_epochs}], Average Loss: {avg_loss:.4f}")

    return model


def test(model, test_loader, type='simple', device=config.DEVICE):
    """
    Test function to evaluate autoencoder performance

    Parameters:
    - model: Trained autoencoder model
    - test_loader: DataLoader for test data
    - type: Type of autoencoder
    - device: Computing device

    Returns:
    - Average reconstruction loss
    - Visualization of original and reconstructed images
    """
    model.eval()
    total_loss = 0.0
    reconstruction_loss = torch.nn.MSELoss(reduction='sum')

    # Disable gradient computation for testing
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.to(device).view(data.size(0), -1)

            if type == 'simple':
                reconstructed = model(data)
                loss = reconstruction_loss(reconstructed, data)

            elif type == 'regularized':
                reconstructed = model(data)
                loss = reconstruction_loss(reconstructed, data)

            elif type == 'denoising':
                noisy_data = data * (1 - config.NOISE_FACTOR) + \
                    torch.rand(data.size()).to(device) * config.NOISE_FACTOR
                reconstructed = model(noisy_data)
                loss = reconstruction_loss(reconstructed, data)

            elif type == 'vae':
                _, reconstructed, mu, sigma = model(data)
                recon_loss = reconstruction_loss(reconstructed, data)
                kld_loss = -0.5 * \
                    torch.sum(1 + sigma - mu.pow(2) - sigma.exp())
                loss = recon_loss + kld_loss

            else:
                raise ValueError(f"Unknown autoencoder type: {type}")

            total_loss += loss.item()

    avg_loss = total_loss / len(test_loader.dataset)
    print(f"Test Average Loss: {avg_loss:.4f}")
    visualize_reconstructions(model, test_loader, type, device)

    return avg_loss


if __name__ == "__main__":
    transform = transforms.ToTensor()
    train_dataset = datasets.MNIST(root=config.DATASET_ROOT,
                                   train=True,
                                   download=True,
                                   transform=transform)
    test_dataset = datasets.MNIST(root=config.DATASET_ROOT,
                                  train=False,
                                  download=True,
                                  transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=config.BATCH_SIZE,
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=config.BATCH_SIZE,
                                              shuffle=False)

    model = AutoEncoder(type='simple').to(config.DEVICE)
    trained_model = train(model, train_loader, type='simple')
