import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


class CategoryAutoencoder(nn.Module):
    def __init__(self, input_dim=128, latent_dim=40):
        super(CategoryAutoencoder, self).__init__()
        self.encoder = nn.Linear(input_dim, latent_dim)
        self.decoder = nn.Linear(latent_dim, input_dim)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
    
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


autoencoder = CategoryAutoencoder()
loss_function = AutoencoderLoss(lambda_value=0.5)
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)


def train_autoencoder(model, dataloader, loss_function, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            main_vector, category_vector = data
            optimizer.zero_grad()
            outputs, latents = model(main_vector)
            loss = loss_function(main_vector, outputs, latents, category_vector)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

# Training the model
train_autoencoder(autoencoder, dataloader, loss_function, optimizer, epochs=20)