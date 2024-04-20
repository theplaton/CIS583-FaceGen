from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from PIL import Image
import os
from torch.utils.data import Dataset
import torch


def divide_by_256(x):
    return x / 256

# Define your transformation pipeline
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(divide_by_256)  # scale to 0-1 range
])


class ImageDataset(Dataset):
    def __init__(self, root_dir, transform, idx_list=None):
        self.root_dir = root_dir
        self.transform = transform
        # self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.image_files = sorted([f for f in os.listdir(root_dir) if f.endswith('.jpg')])
        # if idx_list:
        #     # Convert IDs to strings with leading zeros
        #     id_strings = [str(id).zfill(6) for id in idx_list]
        #     # Filter image files by IDs
        #     filtered_image_files = [f for f in self.image_files if any(f.startswith(id_str) for id_str in id_strings)]
        #     self.image_files = filtered_image_files


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
    def __init__(self, data):
        """
        Args:
            data (list of tuples): Each tuple contains two elements:
                - An array or tensor of shape [lat_img_dim]
                - A category vector or tensor of shape [category_dim]
        """
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        main_vector, category_vector = self.data[idx]
        return torch.tensor(main_vector, dtype=torch.float32), torch.tensor(category_vector, dtype=torch.float32)
    
    


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


def generate_full_data_loader(path, num_workers=8):
    dataset = ImageDataset(path, transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=num_workers)
    return data_loader