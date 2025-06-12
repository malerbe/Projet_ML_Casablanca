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
import cv2
from scipy import ndimage

from tqdm import tqdm

from loss_functions.losses import DiceLoss, IoULoss, FocalLoss, CExDL, DLxFL
import preprocessing.preprocessing as preprocessing
import datasets.datasets as datasets
import utils.utils as utils
import models.unets as unets

def calculate_dice_score(pred, target, smooth=1e-6):
    """
    Calcule le Dice Score entre la prédiction et le target
    """
    pred = pred.contiguous().view(-1)
    target = target.contiguous().view(-1)
    
    intersection = torch.sum(pred * target)
    dice = (2. * intersection + smooth) / (torch.sum(pred) + torch.sum(target) + smooth)
    
    return dice

def calculate_precision_recall(pred, target, smooth=1e-6):
    """
    Calcule precision et recall
    """
    pred_binary = (pred > 0.5).float()
    target_binary = (target > 0.5).float()
    
    tp = torch.sum((pred_binary == 1) & (target_binary == 1)).float()
    fp = torch.sum((pred_binary == 1) & (target_binary == 0)).float()
    fn = torch.sum((pred_binary == 0) & (target_binary == 1)).float()
    
    precision = tp / (tp + fp + smooth)
    recall = tp / (tp + fn + smooth)
    
    return precision, recall

def smart_postprocessing(pred1, pred2, pred3, original_image=None):
    """
    Post-processing intelligent basé sur l'analyse des 3 prédictions
    """
    # Convertir en numpy si nécessaire
    if torch.is_tensor(pred1):
        pred1 = pred1.cpu().numpy().squeeze()
        pred2 = pred2.cpu().numpy().squeeze()
        pred3 = pred3.cpu().numpy().squeeze()
    
    # S'assurer qu'on a des masques binaires
    pred1 = (pred1 > 0.5).astype(np.float32)
    pred2 = (pred2 > 0.5).astype(np.float32)
    pred3 = (pred3 > 0.5).astype(np.float32)
    
    # 1. Zones d'accord : où au moins 2/3 modèles sont d'accord
    agreement_mask = (pred1 + pred2 + pred3) >= 2
    
    # 2. Zones de désaccord : analyser plus finement
    disagreement_mask = ((pred1 + pred2 + pred3) == 1)
    
    # 3. Dans les zones de désaccord, prendre le meilleur modèle (92%)
    final_mask = agreement_mask.astype(np.float32)
    final_mask[disagreement_mask] = pred1[disagreement_mask]  # Modèle à 92%
    
    # 4. Post-processing morphologique pour nettoyer
    if np.any(final_mask > 0):  # Seulement s'il y a des pixels positifs
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        final_mask = cv2.morphologyEx(final_mask.astype(np.uint8), cv2.MORPH_CLOSE, kernel)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_OPEN, kernel)
        
        # 5. Supprimer les petites composantes (ajustez le seuil selon vos données)
        labeled, num_features = ndimage.label(final_mask)
        if num_features > 0:
            sizes = ndimage.sum(final_mask, labeled, range(num_features + 1))
            # Seuil adaptatif basé sur la taille de l'image
            min_size = max(50, int(0.001 * final_mask.size))  # Au moins 0.1% de l'image
            mask_sizes = sizes > min_size
            remove_small = mask_sizes[labeled]
            final_mask = final_mask * remove_small
    
    return final_mask.astype(np.float32)

class SmartEnsembleModel(nn.Module):
    """
    Modèle d'ensemble avec post-processing intelligent
    """
    def __init__(self):
        super().__init__()
        print("Initialisation du Smart Ensemble Model")
    
    def forward(self, x):
        """
        x: [B, 4, H, W] où:
        - Canal 0: Image originale
        - Canal 1: Prédiction modèle 1 (92%)
        - Canal 2: Prédiction modèle 2 (90%) 
        - Canal 3: Prédiction modèle 3 (85%)
        """
        pred1 = x[:, 1, :, :]  # Modèle 92%
        pred2 = x[:, 2, :, :]  # Modèle 90%
        pred3 = x[:, 3, :, :]  # Modèle 85%
        original = x[:, 0, :, :]
        
        batch_results = []
        
        for i in range(x.size(0)):
            result = smart_postprocessing(
                pred1[i], pred2[i], pred3[i], original[i]
            )
            batch_results.append(torch.tensor(result, device=x.device))
        
        return torch.stack(batch_results).unsqueeze(1).float()

def test_smart_ensemble(test_loader, device):
    """
    Tester le modèle d'ensemble intelligent
    """
    model = SmartEnsembleModel().to(device)
    model.eval()
    
    all_dice_scores = []
    all_precision = []
    all_recall = []
    
    print("Test du Smart Ensemble Model...")
    
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(tqdm(test_loader)):
            data, target = data.to(device), target.to(device)
            
            # Prédiction avec le modèle d'ensemble
            ensemble_pred = model(data)
            
            # Calculer les métriques pour chaque image du batch
            for i in range(data.size(0)):
                pred = ensemble_pred[i]
                true = target[i]
                
                # Calculer le Dice score
                dice = calculate_dice_score(pred, true)
                all_dice_scores.append(dice.item())
                
                # Calculer precision et recall
                precision, recall = calculate_precision_recall(pred, true)
                all_precision.append(precision.item())
                all_recall.append(recall.item())
    
    results = {
        'eval_dice_score': np.mean(all_dice_scores),
        'eval_precision': np.mean(all_precision),
        'eval_recall': np.mean(all_recall),
        'eval_dice_std': np.std(all_dice_scores)
    }
    
    return results

def compare_with_individual_models(test_loader, device):
    """
    Comparer l'ensemble avec les modèles individuels
    """
    print("=== COMPARAISON AVEC LES MODÈLES INDIVIDUELS ===")
    
    # Test des modèles individuels
    individual_results = {}
    
    with torch.no_grad():
        all_dice_model1, all_dice_model2, all_dice_model3 = [], [], []
        
        for data, target in tqdm(test_loader, desc="Test modèles individuels"):
            data, target = data.to(device), target.to(device)
            
            pred1 = data[:, 1:2, :, :]  # Modèle 1
            pred2 = data[:, 2:3, :, :]  # Modèle 2
            pred3 = data[:, 3:4, :, :]  # Modèle 3
            
            for i in range(data.size(0)):
                dice1 = calculate_dice_score(pred1[i], target[i])
                dice2 = calculate_dice_score(pred2[i], target[i])
                dice3 = calculate_dice_score(pred3[i], target[i])
                
                all_dice_model1.append(dice1.item())
                all_dice_model2.append(dice2.item())
                all_dice_model3.append(dice3.item())
    
    individual_results = {
        'Model 1 (92%)': np.mean(all_dice_model1),
        'Model 2 (90%)': np.mean(all_dice_model2),
        'Model 3 (85%)': np.mean(all_dice_model3)
    }
    
    # Test de l'ensemble
    ensemble_results = test_smart_ensemble(test_loader, device)
    
    print("\n=== RÉSULTATS ===")
    for model_name, dice in individual_results.items():
        print(f"{model_name:15} | Dice: {dice:.4f}")
    
    print(f"{'Smart Ensemble':15} | Dice: {ensemble_results['eval_dice_score']:.4f} ± {ensemble_results['eval_dice_std']:.4f}")
    print(f"{'':15} | Precision: {ensemble_results['eval_precision']:.4f}")
    print(f"{'':15} | Recall: {ensemble_results['eval_recall']:.4f}")
    
    improvement = ensemble_results['eval_dice_score'] - max(individual_results.values())
    print(f"\n📈 Amélioration: +{improvement:.3f} points de Dice score!")
    
    return ensemble_results

def test_different_strategies(test_loader, device):
    """
    Tester différentes stratégies de combinaison
    """
    print("\n=== TEST DE DIFFÉRENTES STRATÉGIES ===")
    
    strategies = {
        'Majorité (2/3)': lambda p1, p2, p3: (p1 + p2 + p3) >= 2,
        'Unanimité (3/3)': lambda p1, p2, p3: (p1 + p2 + p3) >= 3,
        'Au moins 1 (1/3)': lambda p1, p2, p3: (p1 + p2 + p3) >= 1,
        'Meilleur modèle': lambda p1, p2, p3: p1,  # Juste le modèle à 92%
        'Pondéré': lambda p1, p2, p3: (0.5*p1 + 0.3*p2 + 0.2*p3) > 0.5
    }
    
    strategy_results = {}
    
    for strategy_name, strategy_func in strategies.items():
        print(f"Test de la stratégie: {strategy_name}")
        
        all_dice = []
        
        with torch.no_grad():
            for data, target in tqdm(test_loader, desc=f"Test {strategy_name}"):
                data, target = data.to(device), target.to(device)
                
                pred1 = (data[:, 1, :, :] > 0.5).float()
                pred2 = (data[:, 2, :, :] > 0.5).float()
                pred3 = (data[:, 3, :, :] > 0.5).float()
                
                for i in range(data.size(0)):
                    ensemble_pred = strategy_func(pred1[i], pred2[i], pred3[i]).float()
                    dice = calculate_dice_score(ensemble_pred, target[i])
                    all_dice.append(dice.item())
        
        strategy_results[strategy_name] = np.mean(all_dice)
    
    print("\n=== COMPARAISON DES STRATÉGIES ===")
    for strategy, dice in sorted(strategy_results.items(), key=lambda x: x[1], reverse=True):
        print(f"{strategy:20} | Dice: {dice:.4f}")
    
    return strategy_results

def get_datasets(**params):
    """Obtenir les datasets d'entraînement et de validation"""
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

    train_dataset = datasets.MaskClassificationDataset(
        train_imgs_dir, train_masks_model0_dir, train_masks_model1_dir, 
        train_masks_model2_dir, train_masks_dir, train_transforms
    )
    test_dataset = datasets.MaskClassificationDataset(
        eval_imgs_dir, eval_masks_model0_dir, eval_masks_model1_dir, 
        eval_masks_model2_dir, eval_masks_dir, test_transforms
    )

    return train_dataset, test_dataset

if __name__ == "__main__":
    # Chemins des données
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
    print(f"Utilisation du device: {DEVICE}")
    
    # Paramètres
    params = {
        "IMAGE_WIDTH_HEIGHT": [128, 128],
        "BATCH_SIZE": 16,
        "DEVICE": DEVICE,
        
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
    
    print(f"Dataset de test: {len(test_dataset)} images")
    
    # 1. Tester différentes stratégies simples
    strategy_results = test_different_strategies(test_loader, DEVICE)
    
    # 2. Tester le modèle intelligent avec post-processing
    print("\n" + "="*50)
    ensemble_results = compare_with_individual_models(test_loader, DEVICE)
    
    print(f"\n🎯 Résultat final: {ensemble_results['eval_dice_score']:.1f}% de Dice score!")
