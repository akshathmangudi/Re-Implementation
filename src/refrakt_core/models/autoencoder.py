import torch
import torch.nn as nn
from refrakt_core.registry.model_registry import register_model
from refrakt_core.models.templates.models import BaseAutoEncoder

@register_model("autoencoder")
class AutoEncoder(BaseAutoEncoder):
    def __init__(
        self, input_dim=784, hidden_dim=8, type="simple", model_name="autoencoder"
    ):
        super(AutoEncoder, self).__init__(hidden_dim=hidden_dim, model_name=model_name)
        self.type = type
        self.input_dim = input_dim

        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True),
        )

        self.decoder_layers = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim),
            nn.Sigmoid(),
        )

        if self.type == "vae":
            self.mu = nn.Linear(hidden_dim, hidden_dim)
            self.sigma = nn.Linear(hidden_dim, hidden_dim)

    def encode(self, x):
        encoded = self.encoder_layers(x)
        if self.type == "vae":
            mu = self.mu(encoded)
            sigma = self.sigma(encoded)
            return mu, sigma
        return encoded

    def decode(self, z):
        return self.decoder_layers(z)

    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def training_step(self, batch, optimizer, loss_fn, device):
        """Custom autoencoder training (reconstruction loss)"""
        inputs = batch[0].to(device)
        optimizer.zero_grad()
        recon = self(inputs)
        loss = loss_fn(recon, inputs)
        loss.backward()
        optimizer.step()
        return {"loss": loss.item()}

    def validation_step(self, batch, loss_fn, device):
        """Autoencoder validation (reconstruction loss)"""
        inputs = batch[0].to(device)
        recon = self(inputs)
        loss = loss_fn(recon, inputs)
        return {"val_loss": loss.item()}



    def forward(self, x):
        if self.type in {"simple", "regularized"}:
            encoded = self.encode(x)
            decoded = self.decode(encoded)
            return decoded
        elif self.type == "vae":
            mu, sigma = self.encode(x)
            z = self.reparameterize(mu, sigma)
            decoded = self.decode(z)
            return decoded, mu, sigma
        else:
            raise ValueError(f"Unknown autoencoder type: {self.type}")
