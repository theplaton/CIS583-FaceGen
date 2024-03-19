import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from VAE import VAE

################################################################################
#####################  CREATING DATA LOADERS  ##################################
################################################################################
from ImageLoader import CustomImageDataset

# Define a custom transformation that divides each pixel by 256
divide_by_256 = transforms.Lambda(lambda x: x / 256)

# Define a transform to resize the images, convert them to tensors, and scale to [0, 1]
transform = transforms.Compose([
    # transforms.Resize((144, 158)),
    transforms.ToTensor(),
    divide_by_256           # Divide each pixel by 256
])

# Create the dataset
dataset = CustomImageDataset(root_dir='/Users/Nate/Documents/cs583/midterm/data/img_align_celeba/img_align_celeba', transform=transform)
print("Dataset loaded")
# Split the dataset into train and test sets
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

# Create DataLoaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=True)


################################################################################
##############################  TRAINING  ######################################
################################################################################

vae = VAE(image_channels=3, latent_dim=128)
vae.to()

num_epochs = 10
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
def loss_function(recon_x, x, mu, log_var):
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

log_interval = 1
print("Starting training....")
# Training loop
if __name__ == '__main__':
    freeze_support()
    for epoch in range(num_epochs):
        # Training
        vae.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            # data = data * 2 - 1  # Rescale images from [0, 1] to [-1, 1]
            recon_batch, mu, log_var = vae(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            print(f'batch: {batch_idx} \t{len(train_loader)}')
        
        # Validation
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for data, _ in test_loader:
                data = data * 2 - 1  # Rescale images from [0, 1] to [-1, 1]
                recon_batch, mu, log_var = vae(data)
                loss = loss_function(recon_batch, data, mu, log_var)
                val_loss += loss.item()

    print(f'Epoch: {epoch} \tTraining Loss: {train_loss / len(train_loader):.6f} \tValidation Loss: {val_loss / len(test_loader):.6f}')