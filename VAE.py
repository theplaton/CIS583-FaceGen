import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader



class Encoder(nn.Module):
    def __init__(self, image_channels=3, latent_dim=128):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=4, stride=2, padding=1) # Output: (32, 32, 32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1) # Output: (64, 16, 16)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1) # Output: (128, 8, 8)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1) # Output: (256, 4, 4)
        self.fc_mu = nn.Linear(256*4*4, latent_dim)
        self.fc_logvar = nn.Linear(256*4*4, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        #print(f"shape prior to view : {x.shape}")
        x = x.view(-1, 256*4*4)
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=128, image_channels=3):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256*4*4)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1)

    def forward(self, x):
        x = self.fc(x)
        x = x.view(-1, 256, 4, 4)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = torch.sigmoid(self.conv4(x)) # Use sigmoid to output values between 0 and 1
        x = x.view(-1, 3, 192, 160)
        return x

class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=128):
        super(VAE, self).__init__()
        self.encoder = Encoder(image_channels, latent_dim)
        self.decoder = Decoder(latent_dim, image_channels)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        mu, logvar = self.encoder(x)
        z = self.reparameterize(mu, logvar)
        return self.decoder(z), mu, logvar

# Example usage
# vae = VAE(image_channels=3, latent_dim=128)