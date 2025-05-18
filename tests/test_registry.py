import sys
import os
import pytest
from models.resnet import ResidualBlock

# Add the root directory to the Python path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Now import the registry
from registry.model_registry import MODEL_REGISTRY, get_model, register_model

# Import the models module to ensure all models are registered
import models

class TestModelRegistry:
    def test_registry_contains_models(self):
        """Test that the registry contains the expected models."""
        expected_models = [
            "autoencoder", 
            "convnext", 
            "resnet", 
            "simclr", 
            "srgan",
            "swin"
        ]
        
        for model_name in expected_models:
            assert model_name in MODEL_REGISTRY, f"Model {model_name} not found in registry"
    
    def test_get_model(self):
        """Test that we can get models from the registry."""
        # Test getting an autoencoder model
        autoencoder_model = get_model("autoencoder", input_dim=784, hidden_dim=8)
        assert autoencoder_model is not None
        assert autoencoder_model.model_name == "autoencoder"
        
        # Test getting a classifier model
        resnet_model = get_model("resnet", block=ResidualBlock, layers=[2, 2, 2, 2])
        assert resnet_model is not None
        assert resnet_model.model_name == "resnet"
    
    def test_model_not_found(self):
        """Test that getting a non-existent model raises ValueError."""
        with pytest.raises(ValueError):
            get_model("non_existent_model")