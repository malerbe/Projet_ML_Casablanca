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

from tqdm import tqdm

from loss_functions.losses import DiceLoss, IoULoss, FocalLoss, CExDL, DLxFL
import preprocessing.preprocessing as preprocessing
import datasets.datasets as datasets
import utils.utils as utils
import models.unets as unets

def get_datasets( **params):
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

def save_checkpoint(state, filename="mycheckpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename + ".pth")

def evaluate(model, loader, device):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    num_overlaps = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

            # print(preds.shape, y.shape)

            # Calculer le nombre de pixels corrects
            num_correct += (preds == y).sum().item()
            num_pixels += torch.numel(preds)

            # Calculer le Dice Score
            intersection = (preds * y).sum()
            dice_score += (2 * intersection) / ((preds + y).sum() + 1e-8)

            # Calculer le nombre de prédictions avec au moins un pixel chevauchant le masque réel
            batch_overlaps = ((preds * y).sum(dim=[1, 2, 3]) > 0).sum().item()
            num_overlaps += batch_overlaps

            # Calculer la sensibilité (recall) et la précision (precision)
            true_positives += intersection.item()
            false_positives += (preds * (1 - y)).sum().item()
            false_negatives += ((1 - preds) * y).sum().item()

    accuracy = num_correct / num_pixels * 100
    dice_score = dice_score / len(loader)
    overlap_percentage = num_overlaps / len(loader.dataset) * 100
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)

    # print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}%")

    results = {
        "eval_dice_score": dice_score,
        "eval_overlap_perc": overlap_percentage,
        "eval_precision": precision,
        "eval_recall": recall
    }

    return results

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    import os
    os.makedirs(folder, exist_ok=True)
    
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Premier canal seulement + superposition
        combined_image = (x[:, 0:1, :, :].cpu() * 0.5 + 
                         y.cpu() * 0.5 + 
                         preds.cpu() * 0.5)

        torchvision.utils.save_image(combined_image, f"{folder}/combined_{idx}.png")
        
        if idx >= 100:
            break

def fit(epochs, lr, model, train_loader, eval_loader, device, loss_fct, optimizer, scaler, scheduler, savename):
    losses = []

    model.train()

    for epoch in range(epochs):
        loop = tqdm(train_loader)
        for batch_idx, (data, targets) in enumerate(loop):
            data = data.to(device=device)
            targets = targets.float().to(device=device)
            
            # forward
            with torch.amp.autocast('cuda'):
                predictions = model(data)
                loss = loss_fct(predictions, targets)

            # backward
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # update tqdm loop
            loop.set_postfix(loss=loss.item)

        # Step the scheduler
        scheduler.step()

        # Validation:
        model.eval()
        result = evaluate(model, eval_loader, device)
        print(result)
        losses.append(result)

        # Save every x epochs:
        if epoch%3 == 0:
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            save_checkpoint(checkpoint, filename=savename)
            # print some examples to a folder
            save_predictions_as_imgs(
                eval_loader, model, folder="Cleaned/saved_images/", device=device
            )
        model.train()

    return result

import matplotlib.pyplot as plt

def train_unet(**params):
    # Initialize parameters:
    model = params["MODEL"]


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

    savename = params["SAVENAME"]
   

    batch_size = params["BATCH_SIZE"]
    epochs = params["EPOCHS"]
    loss_fct = params["LOSS_FUNCTION"]
    scaler = params["SCALER"]

    optimizer = params["OPTIMIZER"]

    device = params["DEVICE"]
    print(f"Starting training on device: {device}")

    # 1. Get the train and test dataframes
    train_dataset, test_dataset = get_datasets(**params)

    # 2. Make loaders:
    train_loader, eval_loader = DataLoader(train_dataset, batch_size, shuffle=True), DataLoader(test_dataset, batch_size, shuffle=True)

    images, masks = next(iter(train_loader))

    import matplotlib.pyplot as plt

    # Une batch du loader
    # (Supposons que chaque batch est de la forme (images, masques))
    images, masks = next(iter(train_loader))

    # Affichage de 4 exemples (adapte n)
    n = 4  # nombre d'exemples à afficher
    plt.figure(figsize=(12, 6))

    # 3. Export model to device:
    model.to(device=device)

    # 4. Initialize the learning rate scheduler
    if params["SCHEDULER"] == "step_lr":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # 5. Train model:
    results = fit(epochs, params['LEARNING_RATE'], model, train_loader, eval_loader, device, loss_fct, optimizer, scaler, scheduler, savename)


if __name__ == "__main__":
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

    dice_loss = DiceLoss()
    cross_entropy_loss = nn.CrossEntropyLoss()
    iou_loss = IoULoss()
    focal_loss = FocalLoss()
    CExDL05 = CExDL(alpha=0.5)
    CExDL025 = CExDL(alpha=0.25)
    CExDL075 = CExDL(alpha=0.75)
    DLxFL05 = DLxFL(alpha=0.5)
    DLxFL025 = DLxFL(alpha=0.25)
    DLxFL075 = DLxFL(alpha=0.75)

    
    params = {
        "IMAGE_WIDTH_HEIGHT" : [128, 128],
        "ROTATE_LIMIT": 35,
        "BATCH_NORM": True,
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
        "LEARNING_RATE": 1e-4,
        "DEVICE": DEVICE,
        "BATCH_SIZE": 16,
        "EPOCHS": 100,
        "LOAD_MODEL": False,
        "LOSS_FUNCTION": DLxFL075,
        "KERNEL_POOL_SIZE": 2,
        "CONV_KERNEL_SIZE": 2,
        "DEEP_SUPERVISION": False,
        "SCHEDULER": "step_lr",
        "UNET_FEATURES": [64, 128, 256, 512, 1024],
        
        "SCALER": torch.amp.GradScaler(),
        "SAVENAME": "CLUSTER0.pth.tar",
    }

    model = unets.NestedUNet(in_channels=4, out_channels=1, nb_filter=params["UNET_FEATURES"], deep_supervision=params["DEEP_SUPERVISION"])  
    
    params["MODEL"] = model    
    params["OPTIMIZER"] = optim.Adam(model.parameters(), lr=params["LEARNING_RATE"])

    train_unet(**params)

