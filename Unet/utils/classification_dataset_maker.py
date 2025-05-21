import os
import numpy as np
import pandas as pd
from PIL import Image

# Chemins des dossiers contenant les images et les masques

TRAIN_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\train_images"
TRAIN_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\train_masks"
VAL_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\val_images"
VAL_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\val_masks"


# TRAIN IMAGES:
# ============
# Initialiser les listes pour stocker les noms des images et les tailles
image_names = []
sizes = []

# Lire les noms des fichiers dans le dossier des images
image_files = [f for f in os.listdir(TRAIN_IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    # Charger l'image et le masque
    image_path = os.path.join(TRAIN_IMG_DIR, image_file)
    mask_path = os.path.join(TRAIN_MASKS_DIR, image_file)

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Convertir le masque en tableau numpy
    mask_array = np.array(mask)

    # Calculer les coordonnées minimales et maximales du masque
    mask_coords = np.argwhere(mask_array > 0)
    if mask_coords.size > 0:
        (y_min, x_min), (y_max, x_max) = mask_coords.min(0), mask_coords.max(0) + 1

        # Calculer la taille
        size = (x_max - x_min) * (y_max - y_min)
        sizes.append(size)

        # Ajouter le nom de l'image
        image_names.append(image_file)

# TRAIN IMAGES:
# ============
# Initialiser les listes pour stocker les noms des images et les tailles

# Lire les noms des fichiers dans le dossier des images
image_files = [f for f in os.listdir(VAL_IMG_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))]

for image_file in image_files:
    # Charger l'image et le masque
    image_path = os.path.join(VAL_IMG_DIR, image_file)
    mask_path = os.path.join(VAL_MASKS_DIR, image_file)

    image = Image.open(image_path)
    mask = Image.open(mask_path)

    # Convertir le masque en tableau numpy
    mask_array = np.array(mask)

    # Calculer les coordonnées minimales et maximales du masque
    mask_coords = np.argwhere(mask_array > 0)
    if mask_coords.size > 0:
        (y_min, x_min), (y_max, x_max) = mask_coords.min(0), mask_coords.max(0) + 1

        # Calculer la taille
        size = (x_max - x_min) * (y_max - y_min)
        sizes.append(size)

        # Ajouter le nom de l'image
        image_names.append(image_file)


# Créer un DataFrame Pandas
data = {'image_name': image_names, 'size': sizes}
df = pd.DataFrame(data).sort_values('image_name')

# Sauvegarder le DataFrame en tant que fichier CSV
csv_file = r'Unet\data\image_sizes.csv'
df.to_csv(csv_file, index=False)

print(f"Fichier CSV créé avec succès : {csv_file}")
