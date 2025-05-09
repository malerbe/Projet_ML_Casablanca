import torch
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import sys

############################################
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder_path)
############################################

# Importer les modules nécessaires
import datasets
import models.unet as unet

# Charger le modèle entraîné
model = unet.UNet(inchannels=1, numclasses=1)
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Définir les transformations
transform = transforms.Compose([
    transforms.Resize((256, 256)),  # Redimensionner
    transforms.ToTensor()  # Convertir en tensor
])

# Charger l'ensemble de données
train_dataset, eval_dataset = datasets.make_BrainTumorSegDataset(
    r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\images",
    r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\masks",
    transform=transform
)

# Fonction pour afficher les images avant et après la transformation et les statistiques
def display_image_and_resize(image_path, transform):
    # Charger l'image originale
    original_image = Image.open(image_path)
    print("Original image shape:", original_image.size)
    print("Original image mode:", original_image.mode)
    print("Original image format:", original_image.format)
    print("Original image size in bytes:", os.path.getsize(image_path))

    # Appliquer la transformation
    resized_image = transform(original_image)
    print("Resized image shape:", resized_image.shape)

    # Convertir les images en numpy pour l'affichage
    original_image = np.array(original_image)
    resized_image = resized_image.numpy().transpose((1, 2, 0))  # Transposer pour obtenir (H, W, C)

    # Afficher les images
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title('Original Image')
    ax[0].axis('off')

    ax[1].imshow(resized_image, cmap='gray')
    ax[1].set_title('Resized Image')
    ax[1].axis('off')

    plt.show()

# Chemin de l'image à afficher
index = 0
image_path = eval_dataset.img_paths[index]

# Afficher l'image avant et après la transformation et les statistiques
display_image_and_resize(image_path, transform)
