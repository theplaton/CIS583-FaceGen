from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset
import torch
import linecache
from utils import resolve_device


def divide_by_256(x):
    return x / 256

# Define your transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(divide_by_256)  # scale to 0-1 range
])


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # Get the dimensions of the image
        width, height = image.size

        # Define the coordinates of the box to crop
        # (left, upper, right, lower)
        # 178 -> 160
        # 218 -> 192
        crop_box = (9, 13, width - 9, height - 13)

        # Crop the image
        cropped_image = image.crop(crop_box)

        if self.transform:
            final_image = self.transform(cropped_image)

        return final_image


class LatentFacesWithCategories(Dataset):
    def __init__(self, latent_img_dir, category_attr_path, mu_scale=-100):
        self.image_dir = latent_img_dir
        self.category_attr_file = category_attr_path
        self.data_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.dat')])
        self.mu_scale = mu_scale
    
    def read_category_attribute(self, idx):
        line = linecache.getline(self.category_attr_file, idx + 3) # +3 to skip column names
        parts = line.split()
        return torch.tensor([float(x) for x in parts[1:]]) # Convert the rest of the columns to integers

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, idx):
        loaded_data = torch.load(os.path.join(self.image_dir, self.data_files[idx]))
        mu, logvar = loaded_data['tensor1'].squeeze(0) * self.mu_scale, loaded_data['tensor2'].squeeze(0)
        data = torch.cat((mu, logvar), dim=0)
        category_vector = self.read_category_attribute(idx)
        category_vector.to("mps")
        return data, category_vector
    
    
class FacesAndAttributes(Dataset):
    def __init__(self, img_dir, category_attr_path, transform, device):
        self.image_dir = img_dir
        self.category_attr_file = category_attr_path
        self.image_files = sorted([f for f in os.listdir(self.image_dir) if f.endswith('.jpg')])
        self.transform = transform
        self.device = device
    
    def read_category_attribute(self, idx):
        line = linecache.getline(self.category_attr_file, idx + 3) # +3 to skip column names
        parts = line.split()
        return torch.tensor([float(x) for x in parts[1:]]) # Convert the rest of the columns to integers

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')

        # Get the dimensions of the image
        width, height = image.size

        # Define the coordinates of the box to crop
        # (left, upper, right, lower)
        # 178 -> 160
        # 218 -> 192
        crop_box = (9, 13, width - 9, height - 13)

        # Crop the image
        cropped_image = image.crop(crop_box)

        if self.transform:
            image_tensor = self.transform(cropped_image)
        image_tensor = image_tensor.to(self.device)

        category_tensor = self.read_category_attribute(idx).to(self.device)
        
        return image_tensor, category_tensor
    

def generate_training_data_loaders(path, batch_size, split_ratio=0.8, num_workers=8, shuffle=True):
    # Create the dataset
    dataset = ImageDataset(path, transform)

    # Split the dataset into train and test sets
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for training and testing with multiple workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    return train_loader, test_loader


def generate_full_data_loader(path, num_workers=2):
    dataset = ImageDataset(path, transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers)
    return data_loader


def generate_training_categoryAE_data_loaders(lat_img_path, category_attr_path, batch_size, split_ratio=0.8, num_workers=8, shuffle=True):
    # Create the dataset
    dataset = LatentFacesWithCategories(lat_img_path, category_attr_path)

    # # Split the dataset into train and test sets
    # train_size = int(split_ratio * len(dataset))
    # test_size = len(dataset) - train_size
    # train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # # Create DataLoaders for training and testing with multiple workers
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    
    # return train_loader, test_loader

    # train_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers)
    train_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True)

    return train_loader


def generate_training_img_and_attrs_data_loaders(imgs_dir_path, attrs_path, batch_size, split_ratio=0.8, num_workers=8, shuffle=True):
    # Create the dataset
    dataset = FacesAndAttributes(imgs_dir_path, attrs_path, transform, resolve_device())

    # Split the dataset into train and test sets
    train_size = int(split_ratio * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoaders for training and testing with multiple workers
    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)

    return train_loader, test_loader


# data = data.to('cpu').pin_memory()

# # Move the tensor to the target device
# data = data.to(device, non_blocking=True)