import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import os
import sys

############################################
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder_path)
############################################

import datasets
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from models.classification_cnn import CNN_Network
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    get_loaders_for_classification,
    check_accuracy,
    check_accuracy_2,
    check_accuracy_classification,
    save_predictions_as_imgs,
)
import itertools
from sklearn.model_selection import ParameterGrid
from loss_functions.losses import DiceLoss, IoULoss, FocalLoss, CExDL, DLxFL

# Fonction d'entraînement
def cnn_trainer(loader, model, optimizer, loss_fct, scaler, scheduler, **args):
    
    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(args["NUM_EPOCHS"]):
        loop = tqdm(loader)
    
        for batch_idx, (data, targets, img_name, size) in enumerate(loop):
            # print(f"data {data.shape[0]}, targets {targets.shape[0]}")
            data = data.to(device=args["DEVICE"])
            targets = targets.to(device=args["DEVICE"]).to(float)
            targets = targets.unsqueeze(1)

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
            loop.set_postfix(loss=loss.item())

        # Step the scheduler
        scheduler.step()

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        if args["LOAD_MODEL"]:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

        check_accuracy_classification(loader, model, device=args["DEVICE"])
        scaler = torch.amp.GradScaler('cuda')

        

    return model

# Chemins des dossiers contenant les images et les masques
TRAIN_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\train_images"
TRAIN_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\train_masks"
VAL_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\val_images"
VAL_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\val_masks"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
csv_file = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\image_sizes.csv"

# Définir les fonctions de perte
cross_entropy_loss = nn.CrossEntropyLoss()
BCEWithLogitsLoss = nn.BCEWithLogitsLoss()

# Paramètres de la grille
parameters_grid = {
    "IMAGE_WIDTH_HEIGHT" : [[128, 128]],
    "ROTATE_LIMIT": [35],
    "BATCH_NORM": [True],
    "TRAIN_IMG_DIR": [TRAIN_IMG_DIR],
    "TRAIN_MASKS_DIR": [TRAIN_MASKS_DIR],
    "VAL_IMG_DIR": [VAL_IMG_DIR],
    "VAL_MASKS_DIR": [VAL_MASKS_DIR],
    "LEARNING_RATE": [1e-2],
    "DEVICE": [DEVICE],
    "BATCH_SIZE": [16],
    "NUM_EPOCHS": [100],
    "LOAD_MODEL": [False],
    "LOSS_FUNCTION": [BCEWithLogitsLoss],
    "KERNEL_POOL_SIZE": [2],
    "CONV_KERNEL_SIZE": [2],
    "SCHEDULER": ["step_lr"],
}

# Générer toutes les combinaisons de paramètres
param_grid = ParameterGrid(parameters_grid)

# Définir une fonction pour évaluer les performances
def evaluate_model(params, train_loader, model, optimizer, loss_fct, scaler, scheduler):
    model = cnn_trainer(train_loader, model, optimizer, loss_fct, scaler, scheduler, **params)

    # Évaluer les performances sur le jeu de validation
    performance = check_accuracy_2(val_loader, model, device=params["DEVICE"])

    return model, performance

# Tester toutes les combinaisons de paramètres
best_performance = float('inf')
best_params = None

if __name__ == "__main__":
    combination_count = 0
    for params in param_grid:
        combination_count += 1
        print(f"Training with parameters: {params} on device: {DEVICE}")
        print(f"Number of combinations tested: {combination_count}/{len(param_grid)}")

        train_transform = A.Compose(
            [
                A.Resize(height=params["IMAGE_WIDTH_HEIGHT"][1], width=params["IMAGE_WIDTH_HEIGHT"][0]),
                A.Rotate(limit=params["ROTATE_LIMIT"], p=0.9),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )

        val_transform = A.Compose(
            [
                A.Resize(height=params["IMAGE_WIDTH_HEIGHT"][1], width=params["IMAGE_WIDTH_HEIGHT"][0]),
                A.Normalize(
                    mean=[0.0, 0.0, 0.0],
                    std=[1.0, 1.0, 1.0],
                    max_pixel_value=255.0,
                ),
                ToTensorV2(),
            ]
        )
    
        params_model={
        "shape_in": (1,params["IMAGE_WIDTH_HEIGHT"][1],params["IMAGE_WIDTH_HEIGHT"][0]), 
        "initial_filters": 64,    
        "num_fc1": 100,
        "dropout_rate": 0.05,
        "num_classes": 1}

        model = CNN_Network(params_model)
        model.to(params["DEVICE"])
        loss_fct = params['LOSS_FUNCTION']
        optimizer = optim.Adam(model.parameters(), lr=params["LEARNING_RATE"])

        # Initialize the learning rate scheduler
        if params["SCHEDULER"] == "step_lr":
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.05)

        train_loader, val_loader = get_loaders_for_classification(
            params["TRAIN_IMG_DIR"],
            params["VAL_IMG_DIR"],
            csv_file,
            params["BATCH_SIZE"],
            train_transform,
            val_transform
        )

        scaler = torch.amp.GradScaler('cuda')

        model, performance = evaluate_model(params, train_loader, model, optimizer, loss_fct, scaler, scheduler)

        # Créer un dossier  sauvegarder les résultats
        results_folder = f"results_{params['IMAGE_WIDTH_HEIGHT']}_{params['ROTATE_LIMIT']}_{params['BATCH_NORM']}_{params['BATCH_SIZE']}_lr_{params['LEARNING_RATE']:.1e}"
        os.makedirs(results_folder, exist_ok=True)

        # Créer un sous-dossier pour sauvegarder les images générées
        saved_images_folder = os.path.join(results_folder, "saved_images")
        os.makedirs(saved_images_folder, exist_ok=True)

        # Sauvegarder le modèle entraîné
        torch.save(model.state_dict(), os.path.join(results_folder, "trained_model.pth"))

        # Sauvegarder des exemples d'images générées
        save_predictions_as_imgs(val_loader, model, folder=os.path.join(results_folder, "saved_images"), device=params["DEVICE"])

        # Sauvegarder les paramètres et les performances dans un fichier texte
        with open(os.path.join(results_folder, "params_and_performance.txt"), "w") as f:
            f.write(f"Parameters: {params}\n")
            f.write(f"Performance: {performance}\n")

    print(f"Best parameters: {best_params}")
    print(f"Best performance: {best_performance}")