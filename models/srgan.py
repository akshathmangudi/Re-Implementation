import math
import torch
import torch.nn as nn
from models.templates.models import BaseGAN
from refrakt.registry.model_registry import register_model
from models.resnet import ResidualBlock as BaseResidualBlock


class UpsampleBlock(nn.Module):
    """
    Optimized upsampling block using ConvTranspose2d.
    """
    def __init__(self, in_channels, out_channels, scale_factor=2):
        super(UpsampleBlock, self).__init__()
        self.upsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels * scale_factor ** 2, kernel_size=3, padding=1),
            nn.PixelShuffle(scale_factor),
            nn.PReLU()
        )

    def forward(self, x):
        return self.upsample(x)


class SRResidualBlock(BaseResidualBlock):
    """
    A modified version of ResidualBlock specifically for Super Resolution.
    Inherits from the base ResidualBlock but adapts it for SR requirements.
    """
    def __init__(self, channels):
        # We use a custom initialization instead of the parent constructor
        # since we need different layer configurations
        nn.Module.__init__(self)  # Initialize as nn.Module instead of calling super()
        
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        # Use PReLU instead of ReLU for better gradient flow in generator
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        res = self.conv1(x)
        res = self.bn1(res)
        res = self.prelu(res)
        res = self.conv2(res)
        res = self.bn2(res)
        return x + res


class Generator(nn.Module):
    def __init__(self, scale_factor=4):
        super(Generator, self).__init__()
        self.scale_factor = scale_factor
        upsample_num = int(math.log(scale_factor, 2))

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4), nn.PReLU()
        )
        self.res_blocks = nn.Sequential(
            SRResidualBlock(64),
            SRResidualBlock(64),
            SRResidualBlock(64),
            SRResidualBlock(64),
            SRResidualBlock(64),
        )

        self.final = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64)
        )

        # Build upsample blocks dynamically based on scale factor
        upsample_blocks = [UpsampleBlock(64, 64) for _ in range(upsample_num)]
        upsample_blocks.append(nn.Conv2d(64, 3, kernel_size=9, padding=4))
        self.upsample_blocks = nn.Sequential(*upsample_blocks)

    def forward(self, x):
        block1 = self.block1(x)
        res_blocks = self.res_blocks(block1)
        final = self.final(res_blocks)
        output = block1 + final
        output = self.upsample_blocks(output)
        return (torch.tanh(output) + 1) / 2


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.disc = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(512, 1024, kernel_size=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(1024, 1, kernel_size=1),
        )

    def forward(self, x):
        batch = x.size(0)
        return torch.sigmoid(self.disc(x).view(batch))

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
        """GAN training step (handles both generator/discriminator)"""
        real_imgs = batch.to(device)
        # Update generator
        optimizer["generator"].zero_grad()
        z = torch.randn(batch, 100).to(device)
        fake_imgs = self.generator(z)
        g_loss = loss_fn["generator"](fake_imgs, real_imgs)
        g_loss.backward()
        optimizer["generator"].step()

        # Update discriminator
        optimizer["discriminator"].zero_grad()
        real_pred = self.discriminator(real_imgs)
        fake_pred = self.discriminator(fake_imgs.detach())
        d_loss = loss_fn["discriminator"](real_pred, fake_pred)
        d_loss.backward()
        optimizer["discriminator"].step()

        return {"g_loss": g_loss.item(), "d_loss": d_loss.item()}
    
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