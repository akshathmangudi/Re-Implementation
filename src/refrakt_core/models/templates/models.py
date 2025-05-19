import torch
from typing import Any, Dict
from abc import abstractmethod
from refrakt_core.models.templates.base import BaseModel


class BaseClassifier(BaseModel):
    """
    Base class for classification models.

    Extends the BaseModel with classifier-specific functionality.

    Attributes:
        num_classes (int): Number of classification classes.
    """

    def __init__(self, num_classes: int, model_name: str = "base_classifier"):
        """
        Initialize the base classifier.

        Args:
            num_classes (int): Number of classification classes.
            model_name (str): Name identifier for the model. Defaults to "base_classifier".
        """
        super(BaseClassifier, self).__init__(
            model_name=model_name, model_type="classifier"
        )
        self.num_classes = num_classes

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Class probabilities.
        """
        return self.predict(x, return_probs=True)


class BaseAutoEncoder(BaseModel):
    """
    Base class for autoencoder models.

    Extends the BaseModel with autoencoder-specific functionality.

    Attributes:
        hidden_dim (int): Dimension of the latent space.
    """

    def __init__(self, hidden_dim: int, model_name: str = "base_autoencoder"):
        """
        Initialize the base autoencoder.

        Args:
            hidden_dim (int): Dimension of the latent space.
            model_name (str): Name identifier for the model. Defaults to "base_autoencoder".
        """
        super(BaseAutoEncoder, self).__init__(
            model_name=model_name, model_type="autoencoder"
        )
        self.hidden_dim = hidden_dim

    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to latent representation.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Latent representation.
        """
        pass

    @abstractmethod
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """
        Decode latent representation to output.

        Args:
            z (torch.Tensor): Latent representation.

        Returns:
            torch.Tensor: Reconstructed output.
        """
        pass

    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get latent representation for input.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Latent representation.
        """
        self.eval()
        with torch.no_grad():
            if x.device != self.device:
                x = x.to(self.device)
            return self.encode(x)


class BaseContrastiveModel(BaseModel):
    """
    Base class for contrastive learning models (SimCLR, MoCo, BYOL, DINO).

    Adds support for projection heads and representation learning without relying on labels.
    """

    def __init__(
        self,
        model_name: str = "base_contrastive",
        backbone_name: str = "resnet",
        proj_dim: int = 128,
    ):
        super().__init__(model_name=model_name, model_type="contrastive")
        self.backbone_name = backbone_name
        self.proj_dim = proj_dim

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass returning normalized projection (z) for contrastive loss.
        """
        pass

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns the backbone features (before projection head).
        """
        raise NotImplementedError("Subclasses must implement encode()")

    def project(self, h: torch.Tensor) -> torch.Tensor:
        """
        Returns the projection head output from backbone features.
        """
        raise NotImplementedError("Subclasses must implement project()")

    def predict(self, x: torch.Tensor, return_embedding: bool = False) -> torch.Tensor:
        """
        Returns projection or raw embedding depending on the flag.
        """
        self.eval()
        with torch.no_grad():
            x = x.to(self.device)
            h = self.encode(x)
            if return_embedding:
                return h
            return self.forward(x)

    def summary(self) -> Dict[str, Any]:
        base = super().summary()
        base.update({"backbone": self.backbone_name, "projection_dim": self.proj_dim})
        return base

import torch
from typing import Dict, Any
from abc import abstractmethod
from refrakt_core.models.templates.base import BaseModel


class BaseGAN(BaseModel):
    """
    Base class for Generative Adversarial Network models.

    Extends the BaseModel with GAN-specific functionality, including generator and discriminator components.

    Attributes:
        generator (torch.nn.Module): The generator component of the GAN.
        discriminator (torch.nn.Module): The discriminator component of the GAN.
    """

    def __init__(self, model_name: str = "base_gan"):
        """
        Initialize the base GAN.

        Args:
            model_name (str): Name identifier for the model. Defaults to "base_gan".
        """
        super(BaseGAN, self).__init__(model_name=model_name, model_type="gan")
        self.generator = None
        self.discriminator = None

    @abstractmethod
    def generate(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Generate data using the generator component.

        Args:
            input_data (torch.Tensor): Input tensor for the generator.

        Returns:
            torch.Tensor: Generated output.
        """
        pass

    @abstractmethod
    def discriminate(self, input_data: torch.Tensor) -> torch.Tensor:
        """
        Discriminate data using the discriminator component.

        Args:
            input_data (torch.Tensor): Input tensor for the discriminator.

        Returns:
            torch.Tensor: Discrimination output (typically probability or score).
        """
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass, typically uses the generator.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Generated output.
        """
        return self.generate(x)

    def save_model(self, path: str) -> None:
        """
        Save model weights to disk.

        Args:
            path (str): Path to save the model.
        """
        model_state = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "generator_state_dict": self.generator.state_dict() if self.generator else None,
            "discriminator_state_dict": self.discriminator.state_dict() if self.discriminator else None,
        }
        torch.save(model_state, path)
        print(f"GAN model saved to {path}")

    def load_model(self, path: str) -> None:
        """
        Load model weights from disk.

        Args:
            path (str): Path to load the model from.
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if self.generator and "generator_state_dict" in checkpoint:
            self.generator.load_state_dict(checkpoint["generator_state_dict"])
            
        if self.discriminator and "discriminator_state_dict" in checkpoint:
            self.discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
            
        self.model_name = checkpoint.get("model_name", self.model_name)
        self.model_type = checkpoint.get("model_type", self.model_type)
        print(f"GAN model loaded from {path}")

    def summary(self) -> Dict[str, Any]:
        """
        Get a summary of the GAN model.

        Returns:
            Dict[str, Any]: Model summary information.
        """
        gen_params = sum(p.numel() for p in self.generator.parameters()) if self.generator else 0
        gen_trainable = sum(p.numel() for p in self.generator.parameters() if p.requires_grad) if self.generator else 0
        
        disc_params = sum(p.numel() for p in self.discriminator.parameters()) if self.discriminator else 0
        disc_trainable = sum(p.numel() for p in self.discriminator.parameters() if p.requires_grad) if self.discriminator else 0
        
        return {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "device": self.device,
            "generator_parameters": gen_params,
            "generator_trainable_parameters": gen_trainable,
            "discriminator_parameters": disc_params,
            "discriminator_trainable_parameters": disc_trainable,
            "total_parameters": gen_params + disc_params,
            "total_trainable_parameters": gen_trainable + disc_trainable,
        }

    def to_device(self, device: torch.device) -> "BaseGAN":
        """
        Move model to specified device.

        Args:
            device (torch.device): Device to move the model to.

        Returns:
            BaseGAN: Self reference for method chaining.
        """
        self.device = device
        if self.generator:
            self.generator.to(device)
        if self.discriminator:
            self.discriminator.to(device)
        return self