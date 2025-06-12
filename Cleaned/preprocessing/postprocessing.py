############################################
import os
import sys

parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder_path)
############################################
import os
import sys
import random
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder_path)

import models.classification_cnn
import models.unets
import preprocessing

def load_models(classification_model_path, segmentation_models_paths, device, image_size):
    """Charge tous les modèles nécessaires"""
    models_dict = {}
    
    # Charger le modèle de classification
    classification_model = models.classification_cnn.ClassificationCNN(
        shape_in=[1, image_size, image_size], 
        num_classes=3, 
        configuration=[64, 128, 256]
    )
    classification_model.load_state_dict(torch.load(classification_model_path, map_location=device))
    classification_model.to(device)
    classification_model.eval()
    models_dict['classification'] = classification_model
    
    # Charger les modèles de segmentation
    for i, model_path in enumerate(segmentation_models_paths):
        seg_model = models.unets.NestedUNet(in_channels=1, out_channels=1, nb_filter=[64, 128, 256, 512, 1024], deep_supervision=False)
        
        # Charger le checkpoint complet
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extraire le state_dict du modèle
        if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
            model_state_dict = checkpoint['state_dict']
        else:
            model_state_dict = checkpoint
            
        seg_model.load_state_dict(model_state_dict)
        seg_model.to(device)
        seg_model.eval()
        models_dict[f'segmentation_{i}'] = seg_model
        print(f"Modèle de segmentation {i} chargé avec succès")
    
    return models_dict

def get_all_images(images_dir, masks_dir):
    """Sélectionne toutes les images avec leurs masques correspondants"""
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    
    image_mask_pairs = []
    for img_file in image_files:
        img_path = os.path.join(images_dir, img_file)
        # Supposer que le masque a le même nom que l'image
        mask_path = os.path.join(masks_dir, img_file)
        if os.path.exists(mask_path):
            image_mask_pairs.append((img_path, mask_path))
        else:
            print(f"Masque non trouvé pour {img_file}")
    
    return image_mask_pairs


def preprocess_image(image_path, image_size):
    """Préprocesse une image pour l'inférence"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('L')
    processed_image = transform(image)
    return processed_image.unsqueeze(0)  # Ajouter dimension batch

def load_mask(mask_path, image_size):
    """Charge et préprocesse un masque"""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    mask = Image.open(mask_path).convert('L')
    processed_mask = transform(mask)
    return processed_mask.squeeze(0)  # Enlever dimension channel

def predict_segmentation(model, image_tensor, device):
    """Effectue la prédiction de segmentation"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = torch.sigmoid(model(image_tensor))
        prediction = (prediction > 0.5).float()
        return prediction.squeeze().cpu()

def classify_image(model, image_tensor, device):
    """Classifie l'image pour déterminer quel modèle de segmentation utiliser"""
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        prediction = model(image_tensor)
        predicted_class = torch.argmax(prediction, dim=1).item()
        return predicted_class

def save_dataset(image_mask_pairs, models_dict, device, image_size, output_dir):
    """Sauvegarde le dataset post-traité"""
    images_dir = os.path.join(output_dir, 'Images')
    model1_masks_dir = os.path.join(output_dir, 'Model1Masks')
    model2_masks_dir = os.path.join(output_dir, 'Model2Masks')
    model3_masks_dir = os.path.join(output_dir, 'Model3Masks')
    original_masks_dir = os.path.join(output_dir, 'OriginalMasks')
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(model1_masks_dir, exist_ok=True)
    os.makedirs(model2_masks_dir, exist_ok=True)
    os.makedirs(model3_masks_dir, exist_ok=True)
    os.makedirs(original_masks_dir, exist_ok=True)
    
    for i, (img_path, mask_path) in enumerate(image_mask_pairs):
        print(f"Traitement de l'image {i+1}/{len(image_mask_pairs)}: {os.path.basename(img_path)}")
        
        # Préprocesser l'image
        image_tensor = preprocess_image(img_path, image_size)
        original_image = Image.open(img_path).convert('L').resize((image_size, image_size))
        
        
        
        # Prédictions des 3 modèles
        model1_pred = predict_segmentation(models_dict['segmentation_0'], image_tensor, device)
        model2_pred = predict_segmentation(models_dict['segmentation_1'], image_tensor, device)
        model3_pred = predict_segmentation(models_dict['segmentation_2'], image_tensor, device)
        
        if i < int(0.80*len(image_mask_pairs)):
            # Sauvegarder l'image originale
            original_image.save(os.path.join(images_dir+r"\train", os.path.basename(img_path)))

            # Sauvegarder les masques prédits
            Image.fromarray((model1_pred.numpy() * 255).astype(np.uint8)).save(os.path.join(model1_masks_dir+r"\train", os.path.basename(img_path)))
            Image.fromarray((model2_pred.numpy() * 255).astype(np.uint8)).save(os.path.join(model2_masks_dir+r"\train", os.path.basename(img_path)))
            Image.fromarray((model3_pred.numpy() * 255).astype(np.uint8)).save(os.path.join(model3_masks_dir+r"\train", os.path.basename(img_path)))
            
            # Sauvegarder le masque original
            original_mask = Image.open(mask_path).convert('L').resize((image_size, image_size))
            original_mask.save(os.path.join(original_masks_dir+r"\train", os.path.basename(mask_path)))
        else:
            # Sauvegarder l'image originale
            original_image.save(os.path.join(images_dir+r"\eval", os.path.basename(img_path)))

            Image.fromarray((model1_pred.numpy() * 255).astype(np.uint8)).save(os.path.join(model1_masks_dir+r"\eval", os.path.basename(img_path)))
            Image.fromarray((model2_pred.numpy() * 255).astype(np.uint8)).save(os.path.join(model2_masks_dir+r"\eval", os.path.basename(img_path)))
            Image.fromarray((model3_pred.numpy() * 255).astype(np.uint8)).save(os.path.join(model3_masks_dir+r"\eval", os.path.basename(img_path)))
            
            # Sauvegarder le masque original
            original_mask = Image.open(mask_path).convert('L').resize((image_size, image_size))
            original_mask.save(os.path.join(original_masks_dir+r"\eval", os.path.basename(mask_path)))
        
        print(f"  -> Image et masques sauvegardés avec succès")

def inference_pipeline():
    """Pipeline principal d'inférence"""
    # Configuration
    params = {
        "images_dir": r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\images",
        "masks_dir": r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\masks",
        "classification_model_path": r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\training_results\training_results_10epochs_32batch\model.pth",
        "segmentation_models_paths": [
            r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\CLUSTER0.pth",
            r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\CLUSTER1.pth",
            r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\CLUSTER2.pth"
        ],
        "output_dir": r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\postprocessed",
        "image_size": 128
    }

    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilisation du device: {device}")
    
    # Charger les modèles
    print("Chargement des modèles...")
    try:
        models_dict = load_models(
            params["classification_model_path"],
            params["segmentation_models_paths"],
            device,
            params["image_size"]
        )
        print("Tous les modèles chargés avec succès!")
    except Exception as e:
        print(f"Erreur lors du chargement des modèles: {e}")
        return
    
    # Sélectionner les images
    print("Sélection des images...")
    image_mask_pairs = get_all_images(
        params["images_dir"],
        params["masks_dir"]
    )

    print(f"Nombre d'images sélectionnées: {len(image_mask_pairs)}")
    
    if len(image_mask_pairs) == 0:
        print("Aucune paire image-masque trouvée!")
        return
    
    # Sauvegarder le dataset post-traité
    print("Sauvegarde du dataset post-traité...")
    save_dataset(
        image_mask_pairs,
        models_dict,
        device,
        params["image_size"],
        params["output_dir"]
    )

if __name__ == "__main__":
    inference_pipeline()
