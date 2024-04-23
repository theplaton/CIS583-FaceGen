import torch
import torch.nn as nn
import torch.nn.functional as F
# from tqdm import tqdm
import time
import os
from torchvision import transforms, datasets
from sklearn.metrics import hamming_loss
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from IPython.display import display
from utils import read_configs, resolve_device, get_model_path_by_name, print_progress_with_time, resolve_config
from data_loaders import generate_training_categoryAE_data_loaders, generate_img_and_attrs_select_data_loader, LatentFacesWithCategories
from categoryAE import CategoryAutoencoder
import combined_model
from IPython.display import display
from matplotlib import pyplot as plt

to_image = transforms.ToPILImage()

def test_category_ae():
    attr_ae_model_name = 'category_ae_21_14-34'
    device = resolve_device()
    attr_ae_model = torch.load(get_model_path_by_name(attr_ae_model_name), map_location=device)
    attr_ae_model.eval()

    lat_img_dataset = resolve_config('latent_faces_data_path_abs')
    attr_dataset = resolve_config('face_attributes_rel')
    

    train_loader, test_loader = generate_training_categoryAE_data_loaders(lat_img_dataset, attr_dataset, 1, num_workers=4, shuffle=False)
    test_imgs_len = len(test_loader)
    start_time = time.time()

    all_val_attr = []
    all_val_pred_attr = []

    for idx, data in enumerate(test_loader):
        lat_face, attrs_vector = data
        lat_face = lat_face.to(device)
        attrs_vector = attrs_vector.to(device)
        all_val_attr.append(attrs_vector)
    
        gen_attrs_vector = attr_ae_model.encoder(lat_face)
        predictions = (gen_attrs_vector >= 0).float()
        all_val_pred_attr.append(predictions)
        
        print_progress_with_time(idx, test_imgs_len, start_time, 100)

    all_val_attr = torch.cat(all_val_attr).cpu().numpy()
    all_val_pred_attr = torch.cat(all_val_pred_attr).cpu().numpy()
    val_hamming_loss = hamming_loss(all_val_attr, all_val_pred_attr)
    
    print(" --- TOTAL LOSS: " + str(val_hamming_loss) + " ---")
    
if __name__ == "__main__":
    
    test_category_ae()