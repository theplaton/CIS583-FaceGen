import torch
import torch.nn as nn
import torch
from data_loader import generate_training_categoryAE_data_loaders


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


def train_autoencoder(model, dataloader, loss_function, optimizer, epochs):
    model.train()
    for epoch in range(epochs):
        for data in dataloader:
            face_vector, category_vector = data
            optimizer.zero_grad()
            outputs, latents = model(face_vector)
            loss = loss_function(face_vector, outputs, latents, category_vector)
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")


if __name__ == "__main__":
    
    lat_img_path = "/Users/platonslynko/Desktop/CS583/latent_faces_data"
    category_attr_path = "/Users/platonslynko/Desktop/CS583/CIS583-FaceGen/data/list_attr_celeba.txt"
    
    autoencoder = CategoryAutoencoder()
    loss_function = AutoencoderLoss(lambda_value=0.5)
    optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.001)
    data_loader = generate_training_categoryAE_data_loaders(lat_img_path, category_attr_path, 1, num_workers=1)
    
    # Training the model
    train_autoencoder(autoencoder, data_loader, loss_function, optimizer, epochs=20)