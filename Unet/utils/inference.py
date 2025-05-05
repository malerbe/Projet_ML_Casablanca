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

# Sélectionner une image et un masque au hasard
index = random.randint(0, len(eval_dataset) - 1)
image, mask = eval_dataset[index]

# Ajouter une dimension batch
image = image.unsqueeze(0).to(device)
mask = mask.unsqueeze(0).to(device)

# Faire une prédiction
with torch.no_grad():
    output = model(image)
    predicted_mask = torch.sigmoid(output)
    predicted_mask = (predicted_mask > 0.5).float()  # Seuillage pour obtenir une prédiction binaire

# Convertir les tensors en images
image = image.squeeze().cpu().numpy()
mask = mask.squeeze().cpu().numpy()
predicted_mask = predicted_mask.squeeze().cpu().numpy()

# Superposer les images et les masques
fig, ax = plt.subplots(1, 3, figsize=(18, 6))

# Afficher l'image d'origine
ax[0].imshow(image, cmap='gray')
ax[0].set_title('Image Originale')
ax[0].axis('off')

# Afficher le masque de vérité terrain
ax[1].imshow(mask, cmap='gray')
ax[1].set_title('Masque de Vérité Terrain')
ax[1].axis('off')

# Afficher le masque prédit
ax[2].imshow(predicted_mask, cmap='gray')
ax[2].set_title('Masque Prédit')
ax[2].axis('off')

# Afficher la superposition
plt.show()