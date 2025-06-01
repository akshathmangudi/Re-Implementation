import pytest
import torch

from refrakt_core.registry.model_registry import (MODEL_REGISTRY, get_model,
                                                  register_model)


class TestModelRegistry:
    def test_registry_contains_models(self):
        """Test that the registry contains the registered models."""
        expected_models = [
            "autoencoder",
            "convnext",
            "resnet18",
            "resnet50",
            "resnet101",
            "resnet152",
            "simclr",
            "srgan",
            "swin",
            "vit"
        ]
        
        for model_name in expected_models:
            assert model_name in MODEL_REGISTRY, f"Model {model_name} not found in registry"
    
    def test_register_new_model(self):
        """Test registering a new model."""
        @register_model("test_model")
        class TestModel(torch.nn.Module):
            def __init__(self, model_name="test_model"):
                super().__init__()
                self.model_name = model_name
                self.linear = torch.nn.Linear(10, 5)
                
            def forward(self, x):
                return self.linear(x)
        
        # Check that the model is now in the registry
        assert "test_model" in MODEL_REGISTRY
        
        # Create an instance using the registry
        model = get_model("test_model")
        assert model.model_name == "test_model"
        
        x = torch.randn(2, 10)
        output = model(x)
        assert output.shape == (2, 5)
    
    def test_get_model_with_args(self):
        """Test getting a model with arguments."""
        model = get_model("resnet18", num_classes=20)
        assert model.num_classes == 20
        
        # Test AutoEncoder with custom dimensions
        model = get_model("autoencoder", input_dim=1000, hidden_dim=32)
        assert model.input_dim == 1000
        assert model.hidden_dim == 32
    
    def test_model_not_found(self):
        """Test that trying to get a non-existent model raises an error."""
        with pytest.raises(ValueError):
            get_model("non_existent_model")