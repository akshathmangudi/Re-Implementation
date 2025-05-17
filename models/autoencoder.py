import torch.nn as nn 
from models.templates.base import BaseAutoEncoder

class AutoEncoder(BaseAutoEncoder):
    def __init__(self, input_dim=784, hidden_dim=8, type='simple', model_name="autoencoder"):
        super(AutoEncoder, self).__init__(hidden_dim=hidden_dim, model_name=model_name)
        self.type = type
        self.input_dim = input_dim
        
        # Encoder for all types
        self.encoder_layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(inplace=True)
        )
        
        # Decoder for all types
        self.decoder_layers = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, input_dim),
            nn.Sigmoid()
        )
        
        # Additional layers for VAE
        if self.type == 'vae':
            self.mu = nn.Linear(hidden_dim, hidden_dim)
            self.sigma = nn.Linear(hidden_dim, hidden_dim)
    
    def encode(self, x):
        encoded = self.encoder_layers(x)
        
        if self.type == 'vae':
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
    
    def forward(self, x):
        if self.type == 'simple' or self.type == 'regularized':
            encoded = self.encode(x)
            decoded = self.decode(encoded)
            return decoded
        
        elif self.type == 'vae':
            mu, sigma = self.encode(x)
            z = self.reparameterize(mu, sigma)
            decoded = self.decode(z)
            return decoded, mu, sigma
        
        else:
            raise ValueError(f"Unknown autoencoder type: {self.type}")