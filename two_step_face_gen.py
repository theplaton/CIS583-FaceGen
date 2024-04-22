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
    # face_vae_model_name = 'cnn_vae_model_sumloss.pth'
    attr_ae_model_name = 'category_ae_21_14-34'
    device = resolve_device()
    # face_vae_model = torch.load(get_model_path_by_name(face_vae_model_name), map_location=device)
    # face_vae_model.eval()
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
    

def generate_faces_from_categories():
    face_vae_model_name = 'combined_model_sum-loss_256-latent_0.04_0.96.pth'
    # face_vae_model_name = 'combined_model_sumloss.pth'
    attr_ae_model_name = 'category_ae_21_14-34'
    device = resolve_device()
    face_vae_model = torch.load(get_model_path_by_name(face_vae_model_name), map_location=device)
    face_vae_model.eval()
    attr_ae_model = torch.load(get_model_path_by_name(attr_ae_model_name), map_location=device)
    attr_ae_model.eval()

    lat_img_dataset = resolve_config('latent_faces_data_path_abs')
    img_dataset = resolve_config('data_path_abs')
    attr_dataset = resolve_config('face_attributes_rel')
    # test_attr = resolve_config('test_categories_rel')

    idx_list = [878, 879, 880]
    
    data_loader = generate_img_and_attrs_select_data_loader(img_dataset, attr_dataset, idx_list=idx_list)
    imgs_len = len(data_loader)
    start_time = time.time()

    # all_val_attr = []
    # all_val_pred_attr = []

    for idx, data in enumerate(data_loader):
        img, attrs_vector = data
        img = img.to(device)
        attrs_vector = attrs_vector.to(device)
        
        gen_lat_faces_vector = attr_ae_model.decoder(attrs_vector)
        
        mu, logvar = LatentFacesWithCategories.desqueeze(gen_lat_faces_vector)

        z = face_vae_model.reparameterize(mu, logvar)
        generated_face = face_vae_model.decoder(z)[0]
        img = to_image(generated_face.to('cpu'))
        
        # display(img)
        plt.imshow(img)
        plt.axis('off')  # Turn off axis numbers
        plt.show()
        
        pass
        # print_progress_with_time(idx, imgs_len, start_time, 100)

    # all_val_attr = torch.cat(all_val_attr).cpu().numpy()
    # all_val_pred_attr = torch.cat(all_val_pred_attr).cpu().numpy()
    # val_hamming_loss = hamming_loss(all_val_attr, all_val_pred_attr)
    
    # print(" --- TOTAL LOSS: " + str(val_hamming_loss) + " ---")


if __name__ == "__main__":
    
    # test_category_ae()
    generate_faces_from_categories()