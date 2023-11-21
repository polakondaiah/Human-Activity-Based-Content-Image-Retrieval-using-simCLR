import json
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models import ResNet50_Weights
from PIL import Image,ImageOps
import numpy as np
import os
import json


def load_image_info(json_path):
    with open(json_path, "r") as f:
        return json.load(f)
    
# Function to create a folder if it does not exist
def create_folder_if_not_exist(folder_name):
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)
        print(f"Folder '{folder_name}' created.")
    else:
        print(f"Folder '{folder_name}' already exists.")
    
# Custom Dataset with SimCLR-style Augmentations
class SimCLRDataset(Dataset):
    def __init__(self, image_labels, root_dir, transform=None):
        self.image_labels = image_labels
        self.root_dir = root_dir
        self.transform = transform
        self.filenames = list(image_labels.keys())

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.filenames[idx])
        image = Image.open(img_name).convert("RGB")

        if self.transform:
            pos_1 = self.transform(image)
            pos_2 = self.transform(image)

        return pos_1, pos_2


# SimCLR Model Definition
class SimCLRNetwork(nn.Module):
    def __init__(self):
        super(SimCLRNetwork, self).__init__()
        self.base_model = models.resnet50(weights=ResNet50_Weights.DEFAULT)
        self.base_model.fc = nn.Identity()  # Remove final fully connected layer

        # Projection Head
        self.projection_head = nn.Sequential(
            nn.Linear(2048, 1024), nn.ReLU(), nn.Linear(1024, 512)
        )

    def forward(self, x):
        x = self.base_model(x)
        x = self.projection_head(x)
        return x


# NT-Xent Loss
class NTXentLoss(nn.Module):
    def __init__(self, temperature, batch_size):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

    def forward(self, z_i, z_j):
        N = 2 * z_i.size(0)  # Adjusting for actual batch size
        z = torch.cat((z_i, z_j), dim=0)

        sim = self.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature
        sim_i_j = torch.diag(sim, z_i.size(0))
        sim_j_i = torch.diag(sim, -z_i.size(0))

        # Debug print
#        print(f"sim_i_j size: {sim_i_j.size()}, sim_j_i size: {sim_j_i.size()}")

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[~torch.eye(N, dtype=bool)].reshape(N, -1)

        labels = torch.zeros(N).to(z.device).long()
        loss = nn.CrossEntropyLoss()(
            torch.cat((positive_samples, negative_samples), dim=1), labels
        )
        return loss


# Function for Data Augmentation
def get_simclr_transform(size):
    # Ensure the kernel size is odd and positive
    kernel_size = int(0.1 * size)
    if kernel_size % 2 == 0:
        kernel_size += 1

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.GaussianBlur(kernel_size=kernel_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    model.train()
    best_val_loss = float('inf')
    for epoch in range(num_epochs):
        total_train_loss = 0

        # Training loop
        for img1, img2 in train_loader:
            img1, img2 = img1.to(device), img2.to(device)
            optimizer.zero_grad()
            z_i = model(img1)
            z_j = model(img2)
            loss = criterion(z_i, z_j)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {avg_train_loss:.4f}")

        # Validation loop
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for img1, img2 in val_loader:
                img1, img2 = img1.to(device), img2.to(device)
                z_i = model(img1)
                z_j = model(img2)
                loss = criterion(z_i, z_j)
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Validation Loss: {avg_val_loss:.4f}")
        
        # Checkpointing
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            # torch.save(model.state_dict(), f"simclr_model_best_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), os.path.join("model_weights", f"simclr_model_best_epoch_{epoch+1}.pth"))

            print(f"Checkpoint saved for epoch {epoch+1} with Validation Loss: {avg_val_loss:.4f}")



# Main Function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load your JSON data
    train_image_info = load_image_info("./data/train_image_info.json")
    validation_image_info = load_image_info("./data/validation_image_info.json")  # Add this

    # Data loading
    transform = get_simclr_transform(224)
    train_dataset = SimCLRDataset(train_image_info, "./data/train/", transform)
    val_dataset = SimCLRDataset(validation_image_info, "./data/validation/", transform)  # Add this

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)  # Add this
    # Usage example
    create_folder_if_not_exist("model_weights")
    # Model, optimizer, and loss function setup
    model = SimCLRNetwork().to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = NTXentLoss(temperature=0.5, batch_size=32).to(device)

    num_epochs = 10
    train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device)  # Updated

    # Save the trained model
    torch.save(model.state_dict(), "./model_weights/simclr_model.pth")

if __name__ == "__main__":
    main()

