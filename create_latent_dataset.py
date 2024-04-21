import torch
import torch.nn as nn
import torch.nn.functional as F
# from tqdm import tqdm
import time
import os
from torchvision import transforms, datasets

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from IPython.display import display
from utils import read_configs, resolve_device, get_model_path_by_name, print_progress_with_time
from data_loader import generate_full_data_loader


# face_vae_model_path = 'cnn_vae_model_1.pth'
face_vae_model_path = 'cnn_vae_model_sumloss.pth'


if __name__ == "__main__":
    device = resolve_device()
    model = torch.load(get_model_path_by_name(face_vae_model_path), map_location=device)
    model.eval()

    original_dataset = read_configs()['data_path_abs']
    # new_latent_faces_dataset = read_configs()['latent_faces_data_path_abs']
    new_latent_faces_dataset = "/Users/platonslynko/Desktop/CS583/latentfaces2"

    data_loader = generate_full_data_loader(original_dataset)
    total_imgs = len(data_loader)
    start_time = time.time()
    for idx, data in enumerate(data_loader):
        data = data.to(device)
    
        mu, logvar = model.encoder(data)
        data_to_write = {
            'mu': mu,
            'logvar': logvar
        }
        filename = f'{idx+1:06}.dat'
        torch.save(data_to_write, os.path.join(new_latent_faces_dataset, filename))
        print_progress_with_time(idx, total_imgs, start_time, 100)
