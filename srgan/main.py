import torch
import torch.nn as nn
from tqdm import tqdm
from torch import optim
from torchvision import datasets
from src.loss import PerceptualLoss
from src.utils import get_transform
from torch.utils.data import DataLoader
from src.model import Generator, Discriminator
from src.config import NUM_EPOCHS, DEVICE, BATCH_SIZE, SCALE_FACTOR, LEARNING_RATE


def train():
    for epoch in range(NUM_EPOCHS):
        # Wrap dataloaders with tqdm
        pbar = tqdm(zip(hr_loader, lr_loader), total=len(hr_loader),
                    desc=f'Epoch {epoch+1}/{NUM_EPOCHS}')

        epoch_g_loss = 0
        epoch_d_loss = 0

        for (hr_imgs, _), (lr_imgs, _) in pbar:
            hr_imgs = hr_imgs.to(DEVICE)
            lr_imgs = lr_imgs.to(DEVICE)

            # Generator Training
            g_optimizer.zero_grad()
            sr_imgs = generator(lr_imgs)

            fake_validity = discriminator(sr_imgs)
            g_adv_loss = adversarial_loss(
                fake_validity, torch.ones_like(fake_validity))

            g_content_loss = content_loss(sr_imgs, hr_imgs)
            g_perceptual_loss = perceptual_loss(sr_imgs, hr_imgs)

            g_loss = g_adv_loss + 1e-3 * g_content_loss + 1e-2 * g_perceptual_loss
            g_loss.backward()
            g_optimizer.step()

            # Discriminator Training
            d_optimizer.zero_grad()

            real_validity = discriminator(hr_imgs)
            real_loss = adversarial_loss(
                real_validity, torch.ones_like(real_validity))

            fake_validity = discriminator(sr_imgs.detach())
            fake_loss = adversarial_loss(
                fake_validity, torch.zeros_like(fake_validity))

            d_loss = (real_loss + fake_loss) / 2
            d_loss.backward()
            d_optimizer.step()

            # Update progress bar
            epoch_g_loss += g_loss.item()
            epoch_d_loss += d_loss.item()
            pbar.set_postfix({
                'G Loss': f'{g_loss.item():.4f}',
                'D Loss': f'{d_loss.item():.4f}'
            })

        # Print epoch summary
        print(f'Epoch [{epoch+1}/{NUM_EPOCHS}], Avg G Loss: {epoch_g_loss/len(hr_loader):.4f}, Avg D Loss: {epoch_d_loss/len(hr_loader):.4f}')

        # Save models periodically
        if (epoch + 1) % 10 == 0:
            torch.save(generator.state_dict(),
                       f'generator_epoch_{epoch+1}.pth')
            torch.save(discriminator.state_dict(),
                       f'discriminator_epoch_{epoch+1}.pth')


if __name__ == "__main__":
    # Load Datasets
    hr_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=get_transform(is_hr=True))
    lr_dataset = datasets.CIFAR10(
        root='./data', train=True, download=True, transform=get_transform(is_hr=False))

    hr_loader = DataLoader(hr_dataset, batch_size=BATCH_SIZE, shuffle=True)
    lr_loader = DataLoader(lr_dataset, batch_size=BATCH_SIZE, shuffle=True)

    generator = Generator(SCALE_FACTOR).to(DEVICE)
    discriminator = Discriminator().to(DEVICE)

    adversarial_loss = nn.BCELoss()
    content_loss = nn.MSELoss()
    perceptual_loss = PerceptualLoss()

    g_optimizer = optim.Adam(generator.parameters(), lr=LEARNING_RATE)
    d_optimizer = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE)
    train()
