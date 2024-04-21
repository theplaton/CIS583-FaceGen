import torch
import torch.nn as nn
import torch
import time
from datetime import datetime
from data_loaders import generate_training_categoryAE_data_loaders
from utils import resolve_device, print_training_progress_with_time, read_configs


class CategoryAutoencoder(nn.Module):
    def __init__(self, input_dim=512, latent_dim=40):
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


def train_autoencoder(model, dataloader, loss_function, optimizer, epochs, device, model_save_path):
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
                outputs, latents = model(face_vector)
                loss = loss_function(face_vector, outputs, latents, category_vector)
                loss.backward()
                optimizer.step()
                print_training_progress_with_time(idx, epoch, data_size, epochs, start_time, 100)
            print("\n")
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}", end = "\r")
    except KeyboardInterrupt:
        print("was interrupted!")
    
    now = datetime.now()
    day = now.strftime("%d")
    hr = now.strftime("%H")
    min = now.strftime("%M")
    torch.save(model, f'{model_save_path}_{day}_{hr}-{min}')


if __name__ == "__main__":
    
    # lat_img_path = "/Users/platonslynko/Desktop/CS583/latent_faces_data"
    lat_img_path = read_configs()['latent_faces_data_path_abs']
    category_attr_path = read_configs()['face_attributes_rel']
    model_save_path = "./models/category_ae"
    
    autoencoder = CategoryAutoencoder()
    loss_function = AutoencoderLoss(lambda_value=0.5)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    train_data_loader, test_data_loader = generate_training_categoryAE_data_loaders(lat_img_path, category_attr_path, batch_size = 32, num_workers=8)
    
    # Training the model
    train_autoencoder(autoencoder, train_data_loader, loss_function, optimizer, epochs=20, device=resolve_device(), model_save_path=model_save_path)