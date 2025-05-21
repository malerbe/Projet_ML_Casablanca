import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os
import numpy as np

class BrainTumorSegDataset(Dataset):
    def __init__(self, img_paths, masks_paths, transform=None):
        self.img_paths = img_paths
        self.masks_paths = masks_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image  = Image.open(self.img_paths[index])
        mask = Image.open(self.masks_paths[index])

        if self.transform is not None:
            image = self.transform(image)
            mask = self.transform(mask)
            
        
        return image, mask
    
class BrainTumorSegDataset_2(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        image  = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) / 255

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask
    

def make_BrainTumorSegDataset(imgs_path, masks_path, transform=None, train = 0.2):
    imgs_paths, masks_paths = [os.path.join(imgs_path, path) for path in os.listdir(imgs_path)], [os.path.join(masks_path, path) for path in os.listdir(masks_path)]

    assert len(imgs_paths) == len(masks_paths)

    train_imgs_paths, train_masks_paths = imgs_paths[:int(len(imgs_paths)*train)], masks_paths[:int(len(masks_paths)*train)]
    eval_imgs_paths, eval_masks_paths = imgs_paths[int(len(imgs_path)*train):], masks_paths[int(len(masks_path)*train):]
    
    train_dataset = BrainTumorSegDataset(train_imgs_paths, train_masks_paths, transform)
    eval_dataset = BrainTumorSegDataset(eval_imgs_paths, eval_masks_paths, transform)

    return train_dataset, eval_dataset

import os
import shutil
import random

def remove_unmatched_images(images_dir, masks_dir):
    # Lister tous les fichiers dans les dossiers images et masques
    image_files = os.listdir(images_dir)
    mask_files = os.listdir(masks_dir)

    # Convertir les listes en ensembles pour une vérification rapide
    image_files_set = set([os.path.splitext(img)[0] for img in image_files])
    mask_files_set = set([os.path.splitext(mask)[0] for mask in mask_files])

    # Supprimer les images qui n'ont pas de masque correspondant
    for img in image_files:
        img_base = os.path.splitext(img)[0]
        if img_base not in mask_files_set:
            os.remove(os.path.join(images_dir, img))
            print(f"Removed image: {img} (no matching mask found)")

def split_data(images_dir, masks_dir, train_size=0.8):
    # Créer les dossiers de destination quoi qu'il arrive
    os.makedirs('train_images', exist_ok=True)
    os.makedirs('train_masks', exist_ok=True)
    os.makedirs('val_images', exist_ok=True)
    os.makedirs('val_masks', exist_ok=True)

    # Lister tous les fichiers dans les dossiers images et masques
    image_files = os.listdir(images_dir)
    mask_files = os.listdir(masks_dir)

    # Convertir les listes en ensembles pour une vérification rapide
    image_files_set = set([os.path.splitext(img)[0] for img in image_files])
    mask_files_set = set([os.path.splitext(mask)[0] for mask in mask_files])

    # Trouver les fichiers communs
    common_files = image_files_set.intersection(mask_files_set)

    if not common_files:
        print("Warning: No matching image and mask files found. No files will be copied.")
        return

    # Filtrer les fichiers pour ne garder que ceux ayant un masque correspondant
    image_files = [img for img in image_files if os.path.splitext(img)[0] in common_files]
    mask_files = [mask for mask in mask_files if os.path.splitext(mask)[0] in common_files]

    # Trier les fichiers pour s'assurer que les images et masques correspondent
    image_files.sort()
    mask_files.sort()

    # Mélanger les fichiers pour une division aléatoire
    combined = list(zip(image_files, mask_files))
    if not combined:
        print("Warning: No valid image-mask pairs found. No files will be copied.")
        return
    random.shuffle(combined)
    image_files[:], mask_files[:] = zip(*combined)

    # Calculer le nombre d'images pour l'entraînement et la validation
    num_train = int(train_size * len(image_files))
    train_images = image_files[:num_train]
    train_masks = mask_files[:num_train]
    val_images = image_files[num_train:]
    val_masks = mask_files[num_train:]

    # Copier les fichiers dans les dossiers appropriés
    for img in train_images:
        shutil.copy(os.path.join(images_dir, img), os.path.join('train_images', img))
    for mask in train_masks:
        shutil.copy(os.path.join(masks_dir, mask), os.path.join('train_masks', mask))
    for img in val_images:
        shutil.copy(os.path.join(images_dir, img), os.path.join('val_images', img))
    for mask in val_masks:
        shutil.copy(os.path.join(masks_dir, mask), os.path.join('val_masks', mask))

    print(f"Train images: {len(train_images)}, Train masks: {len(train_masks)}")
    print(f"Validation images: {len(val_images)}, Validation masks: {len(val_masks)}")
    
# Chemins des dossiers images et masques
images_dir = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\car_data\train_images"
masks_dir = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\car_data\train_masks"

# Diviser les données
#remove_unmatched_images(images_dir, masks_dir)
#split_data(images_dir, masks_dir)


import pandas as pd


class TumorSizeDataset(Dataset):
    def __init__(self, image_dir, csv_file, transform=None, threshold=6000):
        self.image_dir = image_dir
        self.annotations = pd.read_csv(csv_file)
        self.transform = transform
        self.threshold = threshold

        # Vérifier que les noms d'images dans le CSV correspondent aux fichiers dans le répertoire
        self.images = os.listdir(image_dir)
        self.annotations = self.annotations[self.annotations['image_name'].isin(self.images)]

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.annotations.iloc[index, 0])
        image  = np.array(Image.open(img_path).convert("L"), dtype=np.float32)
        size = self.annotations.iloc[index, 1]
        label = 1 if size > self.threshold else 0  # Définir un seuil pour la classification binaire
        

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, label, self.annotations.iloc[index, 0], size
    

# # Chemin vers le répertoire des images et le fichier CSV


# import albumentations as A
# from albumentations.pytorch import ToTensorV2
# train_transform = ToTensorV2()



# image_dir = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\train_images"
# csv_file = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\image_sizes.csv"

# # Création du dataset
# dataset = TumorSizeDataset(image_dir=image_dir, csv_file=csv_file, transform=train_transform)

# # Création du DataLoader
# dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

# # Test du dataset
# for batch_idx, (data, targets, img_name, size) in enumerate(dataloader):
#     print(f"Batch {batch_idx}: data shape: {data.shape}, targets: {targets}")