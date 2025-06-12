############################################
import os
import sys

parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder_path)
############################################
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import pandas as pd
import torch.optim as optim
import torchvision
import numpy as np

from tqdm import tqdm

from loss_functions.losses import DiceLoss, IoULoss, FocalLoss, CExDL, DLxFL
import preprocessing.preprocessing as preprocessing
import datasets.datasets as datasets
import utils.utils as utils
import models.unets as unets

class EnsembleModel(nn.Module):
    """
    Ensemble basé sur les prédictions déjà présentes dans le dataset
    """
    def __init__(self, strategy='weighted_average'):
        super(EnsembleModel, self).__init__()
        
        self.strategy = strategy
        
        # Poids basés sur les performances (92%, 90%, 85%)
        self.weights = torch.tensor([0.92, 0.90, 0.85])
        self.weights = self.weights / self.weights.sum()  # Normaliser
        
        # Module d'attention pour la stratégie adaptive (optionnel)
        if strategy == 'adaptive_attention':
            self.attention_net = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),  # 3 prédictions en entrée
                nn.ReLU(),
                nn.Conv2d(16, 8, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 3, 1),  # 3 poids de sortie
                nn.Softmax(dim=1)
            )
    
    def forward(self, x):
        """
        x: [B, 4, H, W] où:
        - Canal 0: Image originale
        - Canal 1: Prédiction modèle 1 (92%)
        - Canal 2: Prédiction modèle 2 (90%) 
        - Canal 3: Prédiction modèle 3 (85%)
        """
        # Extraire les prédictions des 3 modèles (déjà présentes dans le dataset!)
        pred1 = x[:, 1:2, :, :]  # Canal 1 - Modèle à 92%
        pred2 = x[:, 2:3, :, :]  # Canal 2 - Modèle à 90%
        pred3 = x[:, 3:4, :, :]  # Canal 3 - Modèle à 85%
        
        if self.strategy == 'weighted_average':
            # Moyenne pondérée basée sur les performances
            weights = self.weights.to(x.device)
            ensemble_pred = (pred1 * weights[0] + pred2 * weights[1] + pred3 * weights[2])
            
        elif self.strategy == 'majority_vote':
            # Vote majoritaire
            pred1_binary = (pred1 > 0.5).float()
            pred2_binary = (pred2 > 0.5).float()
            pred3_binary = (pred3 > 0.5).float()
            ensemble_pred = ((pred1_binary + pred2_binary + pred3_binary) >= 2).float()
            
        elif self.strategy == 'max_confidence':
            # Prendre la prédiction avec la plus haute confiance pour chaque pixel
            pred_stack = torch.cat([pred1, pred2, pred3], dim=1)  # [B, 3, H, W]
            confidence_scores = torch.abs(pred_stack - 0.5)  # Distance à l'incertitude (0.5)
            best_pred_indices = torch.argmax(confidence_scores, dim=1, keepdim=True)  # [B, 1, H, W]
            
            # Sélectionner la meilleure prédiction par pixel
            ensemble_pred = torch.gather(pred_stack, 1, best_pred_indices)
            
        elif self.strategy == 'adaptive_attention':
            # Utiliser un réseau d'attention pour pondérer dynamiquement
            preds_concat = torch.cat([pred1, pred2, pred3], dim=1)  # [B, 3, H, W]
            attention_weights = self.attention_net(preds_concat)  # [B, 3, H, W]
            
            pred_stack = torch.stack([pred1, pred2, pred3], dim=1)  # [B, 3, 1, H, W]
            attention_weights = attention_weights.unsqueeze(2)  # [B, 3, 1, H, W]
            
            ensemble_pred = torch.sum(pred_stack * attention_weights, dim=1)  # [B, 1, H, W]
            
        else:  # Par défaut: moyenne simple
            ensemble_pred = (pred1 + pred2 + pred3) / 3.0
            
        return ensemble_pred

def get_datasets(**params):
    """Utiliser exactement le même dataset que pour les U-Nets"""
    train_imgs_dir = params["TRAIN_IMGS_DIR"] 
    eval_imgs_dir = params["VAL_IMGS_DIR"] 

    train_masks_model0_dir = params["TRAIN_MASKS_MODEL0_DIR"] 
    eval_masks_model0_dir = params["VAL_MASKS_MODEL0_DIR"]
    train_masks_model1_dir = params["TRAIN_MASKS_MOD2EL1_DIR"] 
    eval_masks_model1_dir = params["VAL_MASKS_MODEL1_DIR"]
    train_masks_model2_dir = params["TRAIN_MASKS_MODEL2_DIR"]
    eval_masks_model2_dir = params["VAL_MASKS_MODEL2_DIR"]
    
    train_masks_dir = params["TRAIN_MASKS_DIR"]
    eval_masks_dir = params["VAL_MASKS_DIR"]

    train_transforms, test_transforms = preprocessing.get_transformations(img_size=params["IMAGE_WIDTH_HEIGHT"][0])

    train_dataset = datasets.MaskClassificationDataset(train_imgs_dir, train_masks_model0_dir, train_masks_model1_dir, train_masks_model2_dir, train_masks_dir, train_transforms)
    test_dataset = datasets.MaskClassificationDataset(eval_imgs_dir, eval_masks_model0_dir, eval_masks_model1_dir, eval_masks_model2_dir, eval_masks_dir, test_transforms)

    return train_dataset, test_dataset

def evaluate_ensemble(model, loader, device):
    """Évaluation pour modèle d'ensemble"""
    num_correct = 0
    num_pixels = 0
    dice_scores = []
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            
            preds = model(x)
            preds_binary = (preds > 0.5).float()

            # Calculer le Dice Score par image
            for i in range(x.size(0)):
                pred_i = preds_binary[i]
                y_i = y[i]
                
                intersection = (pred_i * y_i).sum()
                union = pred_i.sum() + y_i.sum()
                
                if union > 0:
                    dice = (2 * intersection) / (union + 1e-8)
                    dice_scores.append(dice.item())
                else:
                    dice_scores.append(1.0)  # Parfait si les deux sont vides

            # Métriques globales
            num_correct += (preds_binary == y).sum().item()
            num_pixels += torch.numel(preds_binary)
            
            intersection = (preds_binary * y).sum()
            true_positives += intersection.item()
            false_positives += (preds_binary * (1 - y)).sum().item()
            false_negatives += ((1 - preds_binary) * y).sum().item()

    accuracy = num_correct / num_pixels * 100
    dice_score = np.mean(dice_scores)
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    results = {
        "eval_dice_score": dice_score,
        "eval_precision": precision,
        "eval_recall": recall,
        "eval_accuracy": accuracy
    }

    return results

def compare_ensemble_strategies(test_loader, device):
    """Comparer différentes stratégies d'ensemble"""
    strategies = ['weighted_average', 'majority_vote', 'max_confidence', 'simple_average']
    results = {}
    
    for strategy in strategies:
        print(f"\n=== Évaluation stratégie: {strategy} ===")
        
        ensemble_model = EnsembleModel(strategy=strategy).to(device)
        result = evaluate_ensemble(ensemble_model, test_loader, device)
        results[strategy] = result
        
        print(f"Dice Score: {result['eval_dice_score']:.4f}")
        print(f"Precision: {result['eval_precision']:.4f}")
        print(f"Recall: {result['eval_recall']:.4f}")
    
    return results

def train_ensemble_adaptive(**params):
    """Entraîner seulement la partie adaptive de l'ensemble"""
    device = params["DEVICE"]
    
    # Créer le modèle d'ensemble avec attention adaptive
    model = EnsembleModel(strategy='adaptive_attention').to(device)
    
    # Seuls les paramètres de l'attention sont entraînables
    optimizer = optim.Adam(model.attention_net.parameters(), lr=params["LEARNING_RATE"])
    loss_fct = params["LOSS_FUNCTION"]
    
    # Données d'entraînement
    train_dataset, test_dataset = get_datasets(**params)
    train_loader = DataLoader(train_dataset, params["BATCH_SIZE"], shuffle=True)
    test_loader = DataLoader(test_dataset, params["BATCH_SIZE"], shuffle=False)
    
    # Entraînement court (juste pour l'attention)
    for epoch in range(10):  # Peu d'époques nécessaires
        model.train()
        loop = tqdm(train_loader)
        
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device)
            targets = targets.float().to(device)
            
            optimizer.zero_grad()
            predictions = model(data)
            loss = loss_fct(predictions, targets)
            loss.backward()
            optimizer.step()
            
            loop.set_postfix(loss=loss.item())
        
        # Évaluation
        result = evaluate_ensemble(model, test_loader, device)
        print(f"Epoch {epoch}: Dice = {result['eval_dice_score']:.4f}")

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    import os
    os.makedirs(folder, exist_ok=True)
    
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            preds = model(x)
            preds_binary = (preds > 0.5).float()

        # Sauvegarder: image originale, masque GT, prédiction ensemble
        torchvision.utils.save_image(x[:, 0:1, :, :], f"{folder}/original_{idx}.png")
        torchvision.utils.save_image(y, f"{folder}/gt_{idx}.png")
        torchvision.utils.save_image(preds_binary, f"{folder}/ensemble_{idx}.png")
        
        if idx >= 10:
            break

if __name__ == "__main__":
    # Vos chemins existants...
    TRAIN_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\postprocessed\Images\train"
    VAL_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\postprocessed\Images\eval"
    
    TRAIN_MASKS_MODEL0_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\postprocessed\Model1Masks\train"
    VAL_MASKS_MODEL0_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\postprocessed\Model1Masks\eval"
    TRAIN_MASKS_MOD2EL1_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\postprocessed\Model2Masks\train"
    VAL_MASKS_MODEL1_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\postprocessed\Model2Masks\eval"
    TRAIN_MASKS_MODEL2_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\postprocessed\Model3Masks\train"
    VAL_MASKS_MODEL2_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\postprocessed\Model3Masks\eval"

    TRAIN_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\postprocessed\OriginalMasks\train"
    VAL_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\postprocessed\OriginalMasks\eval"

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Paramètres simplifiés
    params = {
        "IMAGE_WIDTH_HEIGHT": [128, 128],
        "LEARNING_RATE": 1e-5,
        "DEVICE": DEVICE,
        "BATCH_SIZE": 16,
        "LOSS_FUNCTION": DiceLoss(),
        
        "TRAIN_IMGS_DIR": TRAIN_IMG_DIR,
        "VAL_IMGS_DIR": VAL_IMG_DIR,
        "TRAIN_MASKS_MODEL0_DIR": TRAIN_MASKS_MODEL0_DIR,
        "VAL_MASKS_MODEL0_DIR": VAL_MASKS_MODEL0_DIR,
        "TRAIN_MASKS_MOD2EL1_DIR": TRAIN_MASKS_MOD2EL1_DIR,
        "VAL_MASKS_MODEL1_DIR": VAL_MASKS_MODEL1_DIR,
        "TRAIN_MASKS_MODEL2_DIR": TRAIN_MASKS_MODEL2_DIR,
        "VAL_MASKS_MODEL2_DIR": VAL_MASKS_MODEL2_DIR,
        "TRAIN_MASKS_DIR": TRAIN_MASKS_DIR,
        "VAL_MASKS_DIR": VAL_MASKS_DIR,
    }
    
    # Créer le test loader
    _, test_dataset = get_datasets(**params)
    test_loader = DataLoader(test_dataset, params["BATCH_SIZE"], shuffle=False)
    
    # Comparer toutes les stratégies
    print("=== COMPARAISON DES STRATÉGIES D'ENSEMBLE ===")
    results = compare_ensemble_strategies(test_loader, DEVICE)
    
    # Afficher le résumé
    print("\n=== RÉSUMÉ DES RÉSULTATS ===")
    for strategy, result in results.items():
        print(f"{strategy:20} | Dice: {result['eval_dice_score']:.4f} | Precision: {result['eval_precision']:.4f} | Recall: {result['eval_recall']:.4f}")
    
    # Entraîner l'attention adaptive
    print("\n=== ENTRAÎNEMENT ATTENTION ADAPTIVE ===")
    train_ensemble_adaptive(**params)
