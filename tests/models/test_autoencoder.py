import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.autoencoder import AutoEncoder

class TestAutoEncoder:
    @pytest.fixture
    def simple_autoencoder(self):
        """Create a simple autoencoder."""
        return AutoEncoder(input_dim=784, hidden_dim=8, type="simple")
    
    @pytest.fixture
    def vae_autoencoder(self):
        """Create a variational autoencoder."""
        return AutoEncoder(input_dim=784, hidden_dim=8, type="vae")
    
    def test_simple_autoencoder_init(self, simple_autoencoder):
        """Test that the simple autoencoder initializes correctly."""
        assert simple_autoencoder.input_dim == 784
        assert simple_autoencoder.hidden_dim == 8
        assert simple_autoencoder.type == "simple"
        assert simple_autoencoder.model_name == "autoencoder"
        assert simple_autoencoder.model_type == "autoencoder"
    
    def test_vae_init(self, vae_autoencoder):
        """Test that the variational autoencoder initializes correctly."""
        assert vae_autoencoder.input_dim == 784
        assert vae_autoencoder.hidden_dim == 8
        assert vae_autoencoder.type == "vae"
        assert hasattr(vae_autoencoder, "mu")
        assert hasattr(vae_autoencoder, "sigma")
    
    def test_simple_encoder(self, simple_autoencoder):
        """Test the encoder for a simple autoencoder."""
        x = torch.randn(10, 784)
        encoded = simple_autoencoder.encode(x)
        assert encoded.shape == (10, 8)
    
    def test_vae_encoder(self, vae_autoencoder):
        """Test the encoder for a variational autoencoder."""
        x = torch.randn(10, 784)
        mu, sigma = vae_autoencoder.encode(x)
        assert mu.shape == (10, 8)
        assert sigma.shape == (10, 8)
    
    def test_reparameterize(self, vae_autoencoder):
        """Test the reparameterization trick for variational autoencoder."""
        mu = torch.zeros(10, 8)
        sigma = torch.zeros(10, 8)
        z = vae_autoencoder.reparameterize(mu, sigma)
        assert z.shape == (10, 8)

        # When sigma is 0 and mu is 0, z should be sampled from N(0, 1)
        # We can't test the exact values due to randomness, but we can check statistics
        # should be roughly standard normal distribution
        assert torch.abs(z.mean()) < 0.5 
        assert torch.abs(z.std() - 1) < 0.5
    
    def test_decoder(self, simple_autoencoder):
        """Test the decoder."""
        z = torch.randn(10, 8)
        decoded = simple_autoencoder.decode(z)
        assert decoded.shape == (10, 784)
        # Output should be between 0 and 1 due to sigmoid activation
        assert torch.all(decoded >= 0) and torch.all(decoded <= 1)
    
    def test_simple_forward(self, simple_autoencoder):
        """Test the forward pass for a simple autoencoder."""
        x = torch.randn(10, 784)
        output = simple_autoencoder(x)
        assert output.shape == (10, 784)
    
    def test_vae_forward(self, vae_autoencoder):
        """Test the forward pass for a variational autoencoder."""
        x = torch.randn(10, 784)
        output, mu, sigma = vae_autoencoder(x)
        assert output.shape == (10, 784)
        assert mu.shape == (10, 8)
        assert sigma.shape == (10, 8)
    
    def test_get_latent(self, simple_autoencoder):
        """Test the get_latent method."""
        x = torch.randn(10, 784)
        latent = simple_autoencoder.get_latent(x)
        assert latent.shape == (10, 8)
    
    def test_invalid_type(self):
        """Test that an invalid type raises ValueError."""
        with pytest.raises(ValueError):
            model = AutoEncoder(input_dim=784, hidden_dim=8, type="invalid")
            x = torch.randn(10, 784)
            model(x)