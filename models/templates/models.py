import torch
from base import BaseModel
from typing import Any, Dict
from abc import abstractmethod


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
