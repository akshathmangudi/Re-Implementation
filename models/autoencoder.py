import torch
import torch.nn as nn


class AutoEncoder(nn.Module):
    def __init__(self, type='simple'):
        super().__init__()
        self.type = type
        self.hidden = 8

        # Encoder for all types
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(784, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, self.hidden),
            torch.nn.ReLU(inplace=True)
        )

        # Decoder for all types
        self.decoder = torch.nn.Sequential(
            torch.nn.Linear(self.hidden, 64),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(64, 256),
            torch.nn.ReLU(inplace=True),
            torch.nn.Linear(256, 784),
            torch.nn.Sigmoid()
        )

        # Additional layers for VAE
        if self.type == 'vae':
            self.mu = torch.nn.Linear(self.hidden, self.hidden)
            self.sigma = torch.nn.Linear(self.hidden, self.hidden)

    def reparametrize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        encoded = self.encoder(x)

        if self.type == 'simple':
            return self.decoder(encoded)

        elif self.type == 'regularized':
            # You can add regularization logic here if needed
            return self.decoder(encoded)

        elif self.type == 'vae':
            mu = self.mu(encoded)
            sigma = self.sigma(encoded)

            # Reparameterization trick
            eta = self.reparametrize(mu, sigma)
            decoded = self.decoder(eta)
            return encoded, decoded, mu, sigma

        else:
            raise ValueError(f"Unknown autoencoder type: {self.type}")
