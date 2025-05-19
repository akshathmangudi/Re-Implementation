import math
import torch
import torch.nn as nn
from refrakt_core.models.templates.models import BaseGAN
from refrakt_core.registry.model_registry import register_model
from refrakt_core.utils.classes.srgan import Generator, Discriminator


@register_model("srgan")
class SRGAN(BaseGAN):
    """
    Super-Resolution Generative Adversarial Network (SRGAN).
    
    This model combines a generator and discriminator to perform
    super-resolution tasks on images.
    
    Inherits from BaseGAN to maintain consistent architecture with other models.
    """
    
    def __init__(self, scale_factor=4, model_name="srgan"):
        """
        Initialize the SRGAN model.
        
        Args:
            scale_factor (int): The upscaling factor for super-resolution. Defaults to 4.
            model_name (str): Model name. Defaults to "srgan".
        """
        super(SRGAN, self).__init__(model_name=model_name)
        self.scale_factor = scale_factor
        self.generator = Generator(scale_factor=scale_factor)
        self.discriminator = Discriminator()

    def training_step(self, batch, optimizer, loss_fn, device):
        lr = batch["lr"].to(device)
        hr = batch["hr"].to(device)

        # Generator update
        optimizer["generator"].zero_grad()
        sr = self.generator(lr)
        g_loss = loss_fn["generator"](sr, hr)
        g_loss.backward()
        optimizer["generator"].step()

        # Discriminator update
        optimizer["discriminator"].zero_grad()
        real_pred = self.discriminator(hr)
        fake_pred = self.discriminator(sr.detach())

        loss_real = loss_fn["discriminator"](real_pred, target_is_real=True)
        loss_fake = loss_fn["discriminator"](fake_pred, target_is_real=False)
        d_loss = 0.5 * (loss_real + loss_fake)

        d_loss.backward()
        optimizer["discriminator"].step()

        return {
            "g_loss": g_loss.item(),
            "d_loss": d_loss.item()
        }


    
    def generate(self, input_data):
        """
        Generate a super-resolution image from a low-resolution input.
        
        Args:
            input_data (torch.Tensor): Low-resolution input image.
            
        Returns:
            torch.Tensor: Super-resolution output image.
        """
        self.generator.eval()
        with torch.no_grad():
            if input_data.device != self.device:
                input_data = input_data.to(self.device)
            return self.generator(input_data)
    
    def discriminate(self, input_data):
        """
        Discriminate between real and fake images.
        
        Args:
            input_data (torch.Tensor): Input image.
            
        Returns:
            torch.Tensor: Probability that the input is a real image.
        """
        self.discriminator.eval()
        with torch.no_grad():
            if input_data.device != self.device:
                input_data = input_data.to(self.device)
            return self.discriminator(input_data)
    
    def summary(self):
        """
        Get a summary of the SRGAN model including additional SR-specific information.
        
        Returns:
            dict: Model summary information.
        """
        base_summary = super().summary()
        # Add SR-specific information
        base_summary.update({
            "scale_factor": self.scale_factor,
        })
        return base_summary
    
    def save_model(self, path):
        """
        Save model weights to disk with SR-specific attributes.
        
        Args:
            path (str): Path to save the model.
        """
        model_state = {
            "model_name": self.model_name,
            "model_type": self.model_type,
            "scale_factor": self.scale_factor,
            "generator_state_dict": self.generator.state_dict(),
            "discriminator_state_dict": self.discriminator.state_dict(),
        }
        torch.save(model_state, path)
        print(f"SRGAN model saved to {path}")
    
    def load_model(self, path):
        """
        Load model weights from disk including SR-specific attributes.
        
        Args:
            path (str): Path to load the model from.
        """
        super().load_model(path)
        checkpoint = torch.load(path, map_location=self.device)
        self.scale_factor = checkpoint.get("scale_factor", self.scale_factor)