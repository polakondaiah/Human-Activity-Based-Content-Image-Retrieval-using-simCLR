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

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

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


def load_model(model_path, device):
    model = SimCLRNetwork().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def generate_embeddings(model, image_path, transform, device):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        embedding = model(image)
    return embedding.cpu().numpy()

def evaluate_model(model, query_data, gallery_data, transform, device, k_values=[1, 10, 50]):
    query_embeddings = {img: generate_embeddings(model, os.path.join("./data/query_images/", img), transform, device) for img in query_data}
    gallery_embeddings = {img: generate_embeddings(model, os.path.join("./data/gallery_images/", img), transform, device) for img in gallery_data}

    ap_scores = []
    ranks = []
    top_retrievals_with_ranks_scores = {}  # Store retrievals with ranks and scores

    for query_img, query_label in query_data.items():
        distances = []
        for gallery_img, gallery_label in gallery_data.items():
            dist = np.linalg.norm(query_embeddings[query_img] - gallery_embeddings[gallery_img])
            distances.append((gallery_img, dist, gallery_label == query_label))

        distances.sort(key=lambda x: x[1])
        relevant_distances = [d for d in distances if d[2]]

        # Store the top retrieved images with their ranks and scores
        top_retrievals_with_ranks_scores[query_img] = [(rank+1, img, score) for rank, (img, score, _) in enumerate(distances[:k_values[-1]])]

        if relevant_distances:
            first_relevant_rank = distances.index(relevant_distances[0])
            ranks.append(first_relevant_rank + 1)

            ap_score = 0.0
            num_relevant_items = 0
            for k in range(1, len(distances) + 1):
                if distances[k - 1][2]:
                    num_relevant_items += 1
                    ap_score += num_relevant_items / k
            ap_score /= len(relevant_distances)
            ap_scores.append(ap_score)

    mean_ap = np.mean(ap_scores)
    mean_rank = np.mean(ranks)

    print(f"Mean Average Precision: {mean_ap}")
    for k in k_values:
        precision_at_k = np.mean([1 if rank <= k else 0 for rank in ranks])
        print(f"Precision at {k}: {precision_at_k}")
    print(f"Mean Rank: {mean_rank}")

    # Return the top retrievals with ranks and scores
    return top_retrievals_with_ranks_scores

# def visualize_retrieval(query_image_path, retrieved_images_info, output_folder, labels_info, top_n=5):
def visualize_retrieval(query_image_path, retrieved_images_info, output_folder, labels_info, gallery_folder, top_n=5, fixed_size=(224, 224)):
    """
    Saves the visualization of the query image and top N retrieved images with their ranks, scores, and labels to a specified folder.
    
    :param query_image_path: File path of the query image.
    :param retrieved_images_info: List of tuples containing rank, file paths of retrieved images, and their scores.
    :param output_folder: Folder where the visualizations will be saved.
    :param labels_info: Dictionary containing image paths as keys and activity labels as values.
    :param gallery_folder: Path to the folder containing gallery images.
    :param top_n: Number of top images to display. Default is 5.
    :param fixed_size: A tuple of the fixed width and height for all images. """   

    if top_n > len(retrieved_images_info):
        print(f"Warning: top_n is greater than the number of retrieved images. Showing only {len(retrieved_images_info)} images.")
        top_n = len(retrieved_images_info)

    fig, axes = plt.subplots(1, top_n + 1, figsize=(20, 10))

    # Resize and display the query image
    query_img = Image.open(query_image_path)
    # query_img = ImageOps.fit(query_img, fixed_size, Image.ANTIALIAS)
    query_img = ImageOps.fit(query_img, fixed_size, Image.Resampling.LANCZOS)
    query_label = labels_info.get(os.path.basename(query_image_path), "Unknown")
    axes[0].imshow(query_img)
    axes[0].set_title(f"Query\n{query_label}")
    axes[0].axis('off')

    for i, (rank, img_path, score) in enumerate(retrieved_images_info[:top_n]):
        full_img_path = os.path.join(gallery_folder, img_path)
        img_label = labels_info.get(os.path.basename(img_path), "Unknown")
        try:
            img = Image.open(full_img_path)
            # img = ImageOps.fit(img, fixed_size, Image.ANTIALIAS)  # Resize image to fixed size
            query_img = ImageOps.fit(query_img, fixed_size, Image.Resampling.LANCZOS)  # Use Resampling.LANCZOS for antialiasing
            axes[i + 1].imshow(img)
            axes[i + 1].set_title(f"Rank {rank}\nScore: {score:.2f}\nLabel: {img_label}")
            axes[i + 1].axis('off')
        except FileNotFoundError:
            print(f"File not found: {full_img_path}, skipping.")

    plt.tight_layout()
    query_img_name = os.path.basename(query_image_path).split('.')[0]
    output_path = os.path.join(output_folder, f"{query_img_name}_visualization.png")
    plt.savefig(output_path)
    plt.close(fig)





# def visualize_results(top_retrievals_with_scores, query_folder, gallery_folder, output_folder, labels_info, top_n=5):
def visualize_results(top_retrievals_with_scores, query_folder, gallery_folder, output_folder, labels_info, top_n=5, fixed_size=(224, 224)):

    """
    Visualizes and saves the top retrieved images with their ranks, scores, and labels for a set of query images.

    :param top_retrievals_with_scores: Dictionary of query image paths and their top retrieved image paths with ranks and scores.
    :param query_folder: Path to the folder containing query images.
    :param gallery_folder: Path to the folder containing gallery images.
    :param output_folder: Folder where the visualizations will be saved.
    :param labels_info: Dictionary containing image paths as keys and activity labels as values.
    :param top_n: Number of top images to display. Default is 5.
    """
    for query_img, retrieved_info in top_retrievals_with_scores.items():
        query_img_path = os.path.join(query_folder, query_img)
        retrieved_imgs_info = [(rank, img, score) for rank, img, score in retrieved_info[:top_n]]  # Fixed unpacking here
        # visualize_retrieval(query_img_path, retrieved_imgs_info, output_folder, labels_info, top_n)
        # visualize_retrieval(query_img_path, retrieved_imgs_info, output_folder, labels_info, gallery_folder, top_n)
        visualize_retrieval(query_img_path, retrieved_imgs_info, output_folder, labels_info, gallery_folder, top_n, fixed_size)


def test_main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model("./model_weights/simclr_model_best_epoch_10.pth", device)
    results = "results"
    create_folder_if_not_exist(results)

    test_transform = get_simclr_transform(224)
    query_data = load_image_info("./data/query_image_info.json")
    gallery_data = load_image_info("./data/gallery_image_info.json")

    # Now retrieve both image paths and labels
    labels_info = {img: label for img, label in {**query_data, **gallery_data}.items()}

    top_retrievals_with_scores = evaluate_model(model, query_data, gallery_data, test_transform, device)
    visualization_folder = os.path.join(results, "test_visualization")
    create_folder_if_not_exist(visualization_folder)

    # Pass the labels_info to visualize_results
    # visualize_results(top_retrievals_with_scores, "./data/query_images/", "./data/gallery_images/", visualization_folder, labels_info)
     # Pass the labels_info and gallery_folder to visualize_results, along with the fixed_size
    visualize_results(top_retrievals_with_scores, "./data/query_images/", "./data/gallery_images/", visualization_folder, labels_info, fixed_size=(224, 224))

    print("Visualizations saved in:", visualization_folder)

if __name__ == "__main__":
    test_main()
