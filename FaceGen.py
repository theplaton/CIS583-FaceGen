import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import sys
from data_loaders import generate_training_img_and_attrs_data_loaders
from utils import resolve_config
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import datetime
from utils import resolve_device, resolve_config, print_training_progress_with_time



class Encoder(nn.Module):
    def __init__(self, image_channels=3, latent_dim=256):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(image_channels, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) 
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        
        self.fc_mu = nn.Linear(256*12*10, latent_dim)
        self.fc_logvar = nn.Linear(256*12*10, latent_dim)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        #print(f"shape after layer 1 : {x.shape}")
        x = F.relu(self.conv2(x))
        #print(f"shape after layer 2 : {x.shape}")
        x = F.relu(self.conv3(x))
        #print(f"shape after layer 3 : {x.shape}")
        x = F.relu(self.conv4(x))
        #print(f"shape prior to view : {x.shape}")
        x = x.view(-1, 256*12*10)
        #print(f"shape after view : {x.shape}")
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

class Decoder(nn.Module):
    def __init__(self, latent_dim=256, image_channels=3):
        super(Decoder, self).__init__()
        self.fc = nn.Linear(latent_dim, 256*12*10)
        self.conv1 = nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding = 1)
        self.conv2 = nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding = 1)
        self.conv3 = nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding = 1)
        self.conv4 = nn.ConvTranspose2d(32, image_channels, kernel_size=3, stride=2, padding = 1, output_padding = 1)

    def forward(self, x):
        #print(f"Decoder: Intial shape : {x.shape}")
        x = self.fc(x)
        x = x.view(-1, 256, 12, 10)
        #print(f"Decoder: shape after layer view : {x.shape}")
        x = F.relu(self.conv1(x))
        #print(f"Decoder: shape after layer 1 : {x.shape}")
        x = F.relu(self.conv2(x))
        #print(f"Decoder: shape after layer 2 : {x.shape}")
        x = F.relu(self.conv3(x))
        #print(f"Decoder: shape after layer 3 : {x.shape}")
        x = torch.sigmoid(self.conv4(x)) # Use sigmoid to output values between 0 and 1
        #print(f"Decoder: shape prior to view : {x.shape}")
        x = x.view(-1, 3, 192, 160)
        return x

class VAE(nn.Module):
    def __init__(self, image_channels=3, latent_dim=256):
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


class AutoencoderLoss(nn.Module):
    def __init__(self, lambda_value=0.5):
        super(AutoencoderLoss, self).__init__()
        self.lambda_value = lambda_value
        self.reconstruction_loss = nn.MSELoss()
        self.latent_matching_loss = nn.MSELoss()  # Assuming continuous categories

    def forward(self, inputs, reconstructions, latents, categories):
        rec_loss = self.reconstruction_loss(reconstructions, inputs)
        match_loss = self.latent_matching_loss(latents, categories)
        total_loss = rec_loss + self.lambda_value * match_loss
        return total_loss


def train(model, dataloader, loss_function, optimizer, epochs, device, model_safe_path):
    model.to(device)
    model.train()
    data_size = len(dataloader)
    start_time = time.time()
    try:
        for epoch in range(epochs):
            for idx, data in enumerate(dataloader):
                face_vector, category_vector = data
                face_vector = face_vector.to(device)
                category_vector = category_vector.to(device)
                optimizer.zero_grad()
                decoded, outputs, latents = model(face_vector)
                loss = loss_function(face_vector, outputs, latents, category_vector)
                loss.backward()
                optimizer.step()
                print_training_progress_with_time(idx, epoch, data_size, epochs, start_time, 100)
            print("\n")
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")
    except KeyboardInterrupt:
        print("was interrupted!")
    
    now = datetime.now()
    day = now.strftime("%d")
    hr = now.strftime("%H")
    min = now.strftime("%M")
    torch.save(model, f'{model_safe_path}_{day}_{hr}-{min}')


if __name__ == "__main__":
    img_data = resolve_config('data_path_abs')
    attr_data = resolve_config('face_attributes_rel')
    model_safe_path = os.path.join("./models", "face_gen")
    
    autoencoder = VAE()
    loss_function = AutoencoderLoss(lambda_value=0.5)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    
    train_loader, test_loader = generate_training_img_and_attrs_data_loaders(img_data, attr_data, 1)
    train(autoencoder, train_loader, loss_function, optimizer, epochs=10, device=resolve_device(), model_safe_path=model_safe_path)
    