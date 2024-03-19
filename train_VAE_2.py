import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import os
import json
from torchvision import transforms, datasets
import torch
from torchvision import datasets, transforms
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from VAE import VAE
from data_loader import generate_data_loaders


def read_configs():
    # Specify the path to your JSON file
    json_file_path = './config.json'

    # Read the JSON file into a dictionary
    with open(json_file_path, 'r') as file:
        data = json.load(file)
        
    return data


def resolve_device():
    # Check for CUDA and MPS availability
    use_cuda = torch.cuda.is_available()
    use_mps = torch.backends.mps.is_available()

    # Set the device based on availability
    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Print the selected device
    print(f"Using device: {device}")
    
    return device


if __name__ == '__main__':

    device = resolve_device()
    
    ################################################################################
    #####################  CREATING DATA LOADERS  ##################################
    ################################################################################

    config = read_configs()
    batch_size = 32
    # Get datases
    train_loader, test_loader = generate_data_loaders(config['data_path'], batch_size)
    
    ################################################################################
    ##############################  TRAINING  ######################################
    ################################################################################
    
    vae = VAE(image_channels=3, latent_dim=128)
    vae.to(device)
    
    num_epochs = 10
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    def loss_function(recon_x, x, mu, log_var):
        BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        return BCE + KLD
    
    log_interval = 1
    print("Starting training....")
    # Training loop
    
    start_time = time.time()
    current_epoch = 0
    total_batches = len(train_loader)
    
    for epoch in range(num_epochs):
        # Training
        vae.train()
        batches_processed = 0
        train_loss = 0
        current_epoch += 1
        remaining_epochs = num_epochs - current_epoch
        #train_progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}, Train Loss: {0:.6f}", leave = False)
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
        ################################################################################
        ##################### CALCULATING PROCESSING TIME  #############################
        ################################################################################
            batches_processed += 1
            current_time = time.time()
            elapsed_time = current_time - start_time
            processing_rate = batches_processed / elapsed_time
            remaining_batches = total_batches - batches_processed
            total_remaining_batches = remaining_epochs * total_batches + remaining_batches
            total_remaining_time = total_remaining_batches / processing_rate
            print(f"Current loss: {train_loss / (batch_idx+1):8.4f}, Remaining time (s): {total_remaining_time:8.2f}, Remaining batches: {total_remaining_batches:5d}, Remaining_images: {total_remaining_batches*32:5d}, Processing_rate: {processing_rate:6.2f}, BatchID: {batch_idx}", end="\r")
        ################################################################################
            
            # data = data * 2 - 1  # Rescale images from [0, 1] to [-1, 1]
            recon_batch, mu, log_var = vae(data)
            loss = loss_function(recon_batch, data, mu, log_var)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            #print(f'batch: {batch_idx} \t{len(train_loader)}')
            #train_progress_bar.set_description(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss / (batch_idx+1):.6f}")
            #print(f"Current loss: {train_loss / (batch_idx+1)}, Remaining time (s): {remaining_time}", end="\r")
        
        # Validation
        vae.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_idx, data in enumerate(test_loader):
                data = data.to(device)
                #data = data * 2 - 1  #Rescale images from [0, 1] to [-1, 1]
                recon_batch, mu, log_var = vae(data)
                loss = loss_function(recon_batch, data, mu, log_var)
                val_loss += loss.item()
    
        print(f'Epoch: {epoch} \tTraining Loss: {train_loss / len(train_loader):.6f} \tValidation Loss: {val_loss / len(test_loader):.6f}')
