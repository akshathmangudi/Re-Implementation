import pytest
import torch

from refrakt_core.models.resnet import ResidualBlock, ResNet


class TestResNet:
    @pytest.fixture
    def small_resnet(self):
        """Create a small ResNet for testing."""
        return ResNet(
            block=ResidualBlock,
            layers=[1, 1, 1, 1],
            in_channels=3,
            num_classes=10
        )
    
    def test_init(self, small_resnet):
        """Test that the ResNet initializes correctly."""
        assert small_resnet.model_name == "resnet"
        assert small_resnet.model_type == "classifier"
        assert small_resnet.num_classes == 10
    
    def test_residual_block(self):
        """Test the ResidualBlock."""
        block = ResidualBlock(in_channels=64, out_channels=64)
        x = torch.randn(1, 64, 56, 56)
        output = block(x)
        assert output.shape == (1, 64, 56, 56)
        
        # Test with downsample
        block_with_downsample = ResidualBlock(
            in_channels=64,
            out_channels=128,
            stride=2,
            downsample=nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=1, stride=2),
                nn.BatchNorm2d(128)
            )
        )
        output = block_with_downsample(x)
        assert output.shape == (1, 128, 28, 28)
    
    def test_forward(self, small_resnet):
        """Test the forward pass."""
        x = torch.randn(2, 3, 224, 224)
        output = small_resnet(x)
        assert output.shape == (2, 10)
    
    def test_predict(self, small_resnet):
        """Test the predict method."""
        x = torch.randn(2, 3, 224, 224)
        predictions = small_resnet.predict(x)
        assert predictions.shape == (2,)
        assert predictions.dtype == torch.int64 
        
        probabilities = small_resnet.predict(x, return_probs=True)
        assert probabilities.shape == (2, 10)
        assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    
    def test_predict_proba(self, small_resnet):
        """Test the predict_proba method."""
        x = torch.randn(2, 3, 224, 224)
        probabilities = small_resnet.predict_proba(x)
        assert probabilities.shape == (2, 10)
        assert torch.all(probabilities >= 0) and torch.all(probabilities <= 1)
        assert torch.allclose(probabilities.sum(dim=1), torch.ones(2))
    
    def test_model_on_different_input_sizes(self, small_resnet):
        """Test that the model works with different input sizes."""
        x_small = torch.randn(2, 3, 160, 160)
        output_small = small_resnet(x_small)
        assert output_small.shape == (2, 10)
        
        x_large = torch.randn(2, 3, 256, 256)
        output_large = small_resnet(x_large)
        assert output_large.shape == (2, 10)
    
    def test_save_load(self, small_resnet, tmp_path):
        """Test save and load functionality."""
        save_path = tmp_path / "resnet.pt"
        
        x = torch.randn(2, 3, 224, 224)
        original_predictions = small_resnet.predict(x)
        
        # Save the model
        small_resnet.save_model(str(save_path))
        
        # Create a new model and load the saved weights
        new_model = ResNet(
            block=ResidualBlock,
            layers=[1, 1, 1, 1],
            in_channels=3,
            num_classes=10
        )
        new_model.load_model(str(save_path))
        
        loaded_predictions = new_model.predict(x)
        assert torch.all(original_predictions == loaded_predictions)
        
        assert new_model.model_name == small_resnet.model_name
        assert new_model.model_type == small_resnet.model_type
        assert new_model.num_classes == small_resnet.num_classes

import torch.nn as nn  # Add this import for the downsample test
