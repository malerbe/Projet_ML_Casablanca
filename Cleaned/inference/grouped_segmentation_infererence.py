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
import preprocessing.preprocessing as preprocessing

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

def get_random_images(images_dir, masks_dir, num_images=16):
    """Sélectionne des images aléatoirement avec leurs masques correspondants"""
    image_files = [f for f in os.listdir(images_dir) if f.endswith(('.png', '.jpg', '.jpeg'))]
    selected_files = random.sample(image_files, min(num_images, len(image_files)))
    
    image_mask_pairs = []
    for img_file in selected_files:
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
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    image = Image.open(image_path).convert('L')
    processed_image = transform(image)
    return processed_image.unsqueeze(0)  # Ajouter dimension batch

def load_mask(mask_path, image_size):
    """Charge et préprocesse un masque"""
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
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

def create_overlay(image, mask, alpha=0.3, color='red'):
    """Crée une superposition transparente du masque sur l'image"""
    # Convertir l'image en RGB si nécessaire
    if len(image.shape) == 2:
        image_rgb = np.stack([image, image, image], axis=2)
    else:
        image_rgb = image.copy()
    
    # Normaliser les valeurs si nécessaire
    if image_rgb.max() <= 1.0:
        image_rgb = (image_rgb * 255).astype(np.uint8)
    
    # Créer le masque coloré
    mask_colored = np.zeros_like(image_rgb)
    if color == 'red':
        mask_colored[:, :, 0] = mask * 255
    elif color == 'green':
        mask_colored[:, :, 1] = mask * 255
    elif color == 'blue':
        mask_colored[:, :, 2] = mask * 255
    elif color == 'yellow':
        mask_colored[:, :, 0] = mask * 255  # Rouge
        mask_colored[:, :, 1] = mask * 255  # Vert
    
    # Appliquer la transparence
    overlay = image_rgb.copy()
    mask_bool = mask > 0.5
    overlay[mask_bool] = (1 - alpha) * image_rgb[mask_bool] + alpha * mask_colored[mask_bool]
    
    return overlay.astype(np.uint8)

def create_visualization_grid(image_mask_pairs, models_dict, device, image_size, output_path):
    """Crée la grille de visualisation finale"""
    num_images = len(image_mask_pairs)
    fig, axes = plt.subplots(num_images, 5, figsize=(25, 5 * num_images))
    
    if num_images == 1:
        axes = axes.reshape(1, -1)
    
    colors = ['red', 'green', 'blue']  # Couleurs pour les 3 modèles
    
    for i, (img_path, mask_path) in enumerate(image_mask_pairs):
        print(f"Traitement de l'image {i+1}/{num_images}: {os.path.basename(img_path)}")
        
        # Préprocesser l'image
        image_tensor = preprocess_image(img_path, image_size)
        original_image = np.array(Image.open(img_path).convert('L').resize((image_size, image_size)))
        original_mask = load_mask(mask_path, image_size).numpy()
        
        # Classification
        predicted_class = classify_image(models_dict['classification'], image_tensor, device)
        
        # Image originale
        axes[i, 0].imshow(original_image, cmap='gray')
        axes[i, 0].set_title('Image originale', fontsize=10)
        axes[i, 0].axis('off')
        
        # Prédictions des 3 modèles
        for j in range(3):
            seg_pred = predict_segmentation(models_dict[f'segmentation_{j}'], image_tensor, device)
            overlay = create_overlay(original_image, seg_pred.numpy(), alpha=0.4, color=colors[j])
            
            title = f'Modèle {j+1}'
            if j == predicted_class:
                title += ' ⭐ (Sélectionné)'
            
            axes[i, j+1].imshow(overlay)
            axes[i, j+1].set_title(title, fontsize=10)
            axes[i, j+1].axis('off')
        
        # Image avec masque original
        original_overlay = create_overlay(original_image, original_mask, alpha=0.4, color='yellow')
        axes[i, 4].imshow(original_overlay)
        axes[i, 4].set_title('Masque original', fontsize=10)
        axes[i, 4].axis('off')
        
        print(f"  -> Classe prédite: {predicted_class}")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Visualisation sauvegardée : {output_path}")

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
        "output_path": r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\comparison_results.png",
        "image_size": 128,
        "num_images": 16
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
    image_mask_pairs = get_random_images(
        params["images_dir"],
        params["masks_dir"],
        params["num_images"]
    )
    
    print(f"Nombre d'images sélectionnées: {len(image_mask_pairs)}")
    
    if len(image_mask_pairs) == 0:
        print("Aucune paire image-masque trouvée!")
        return
    
    # Créer la visualisation
    print("Création de la visualisation...")
    create_visualization_grid(
        image_mask_pairs,
        models_dict,
        device,
        params["image_size"],
        params["output_path"]
    )

if __name__ == "__main__":
    inference_pipeline()
