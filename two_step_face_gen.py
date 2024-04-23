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

def generate_faces_from_categories():
    face_vae_model_name = 'combined_model_sum-loss_256-latent_0.04_0.96.pth'
    
    # attr_ae_model_name = 'category_ae_22_16-41'
    # attr_ae_model_name = 'category_ae_22_15-55'
    # attr_ae_model_name = 'category_ae_22_00-42'
    # attr_ae_model_name = 'category_ae_21_14-34'
    attr_ae_model_name = 'category_ae_20_22-57'
    
    device = resolve_device()
    face_vae_model = torch.load(get_model_path_by_name(face_vae_model_name), map_location=device)
    face_vae_model.eval()
    attr_ae_model = torch.load(get_model_path_by_name(attr_ae_model_name), map_location=device)
    attr_ae_model.eval()

    lat_img_dataset = resolve_config('latent_faces_data_path_abs')
    img_dataset = resolve_config('data_path_abs')
    attr_dataset = resolve_config('face_attributes_rel')
    generated_faces_path = resolve_config('generated_faces_abs')

    idx_list = [878, 879, 880]
    
    data_loader = generate_img_and_attrs_select_data_loader(img_dataset, attr_dataset, idx_list=idx_list)
    imgs_len = len(data_loader)
    start_time = time.time()

    for idx, data in enumerate(data_loader):
        orig_img, attrs_vector = data
        attrs_vector = attrs_vector.to(device)
        
        gen_lat_faces_vector = attr_ae_model.decoder(attrs_vector)
        
        mu, logvar = LatentFacesWithCategories.desqueeze(gen_lat_faces_vector)

        z = face_vae_model.reparameterize(mu, logvar)
        generated_face = face_vae_model.decoder(z)[0]
        # img = to_image(generated_face.to('cpu'))
        img = generated_face.to('cpu')
        
        img = img.detach().permute(1, 2, 0).numpy()
        orig_img = orig_img[0].permute(1, 2, 0)
        
        fig, ax = plt.subplots()
        
        ax.imshow(orig_img, extent=[-200, 0, -100, 100])
        ax.imshow(img, extent=[0, 200, -100, 100])
        # plt.imshow(img)
        ax.set_xlim([-200, 200])
        ax.set_ylim([-100, 100])
        ax.set_xticks([])
        ax.set_yticks([])
        plt.axis('off')  # Turn off axis numbers
        filename = os.path.join(generated_faces_path, f'{idx+1:06}.png')
        plt.savefig(filename, bbox_inches='tight')
        # plt.show()

        print_progress_with_time(idx, imgs_len, start_time, 100)




if __name__ == "__main__":
    
    generate_faces_from_categories()