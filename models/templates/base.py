import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Any
from abc import ABC, abstractmethod


class BaseModel(nn.Module, ABC):
    """
    Abstract base class for all neural network models.

    This class provides a common interface for different model architectures,
    including methods for forward pass, prediction, and saving/loading model weights.

    Attributes:
        device (torch.device): Device to run the model on.
        model_name (str): Name identifier for the model.
        model_type (str): Type/architecture of the model.
    """

    def __init__(self, model_name: str = "base_model", model_type: str = "generic"):
        """
        Initialize the base model.

        Args:
            model_name (str): Name identifier for the model. Defaults to "base_model".
            model_type (str): Type/architecture of the model. Defaults to "generic".
        """
        super(BaseModel, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name
        self.model_type = model_type

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        pass

    def predict(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Perform prediction with the model.

        Args:
            x (torch.Tensor): Input tensor.
            **kwargs: Additional arguments for specific model implementations.

        Returns:
            torch.Tensor: Model predictions.
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            if x.device != self.device:
                x = x.to(self.device)
            output = self.forward(x)

            # Handle different output types based on model_type
            if self.model_type == "classifier":
                # For classifiers, return class probabilities or indices
                if kwargs.get("return_probs", False):
                    return torch.softmax(output, dim=1)
                else:
                    return torch.argmax(output, dim=1)
            elif self.model_type == "autoencoder":
                # For autoencoders, return reconstructed output
                return output
            else:
                # Default behavior
                return output

    def save_model(self, path: str) -> None:
        """
        Save model weights to disk.

        Args:
            path (str): Path to save the model.
        """
        model_state = {
            "model_state_dict": self.state_dict(),
            "model_name": self.model_name,
            "model_type": self.model_type,
        }
        torch.save(model_state, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model weights from disk.

        Args:
            path (str): Path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.load_state_dict(checkpoint["model_state_dict"])
        self.model_name = checkpoint.get("model_name", self.model_name)
        self.model_type = checkpoint.get("model_type", self.model_type)
        print(f"Model loaded from {path}")

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the model.

        Returns:
            Dict[str, Any]: Model summary information.
        """
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }

    def to_device(self, device: torch.device) -> "BaseModel":
        """
        Move model to specified device.

        Args:
            device (torch.device): Device to move the model to.

        Returns:
            BaseModel: Self reference for method chaining.
        """
        self.device = device
        return self.to(device)
