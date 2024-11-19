import torch
import config
import matplotlib.pyplot as plt


def visualize_reconstructions(model, test_loader, type, device, num_samples=5):
    """
    Visualize original and reconstructed images

    Parameters:
    - model: Trained autoencoder model
    - test_loader: DataLoader for test data
    - type: Type of autoencoder
    - device: Computing device
    - num_samples: Number of samples to visualize
    """
    model.eval()
    with torch.no_grad():
        # Get first batch
        data, _ = next(iter(test_loader))
        data = data.to(device).view(data.size(0), -1)

        # Reconstruct images based on autoencoder type
        if type == 'simple':
            reconstructed = model(data)
        elif type == 'regularized':
            reconstructed = model(data)
        elif type == 'denoising':
            noisy_data = data * (1 - config.NOISE_FACTOR) + \
                torch.rand(data.size()).to(device) * config.NOISE_FACTOR
            reconstructed = model(noisy_data)
        elif type == 'vae':
            _, reconstructed, _, _ = model(data)

        # Prepare for visualization
        data = data.cpu().view(-1, 28, 28)
        reconstructed = reconstructed.cpu().view(-1, 28, 28)

        # Plot
        plt.figure(figsize=(10, 4))
        for i in range(num_samples):
            # Original images
            plt.subplot(2, num_samples, i + 1)
            plt.imshow(data[i], cmap='gray')
            plt.title('Original')
            plt.axis('off')

            # Reconstructed images
            plt.subplot(2, num_samples, num_samples + i + 1)
            plt.imshow(reconstructed[i], cmap='gray')
            plt.title('Reconstructed')
            plt.axis('off')

        plt.tight_layout()
        plt.savefig(f'{type}_autoencoder_reconstructions.png')
        plt.close()
