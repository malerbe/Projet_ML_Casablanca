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
import models.unet as unet

def evaluate(model, eval_loader, criterion, device):
    model.eval()
    eval_loss = 0.0
    with torch.no_grad():
        for images, masks in eval_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            eval_loss += loss.item()
    return eval_loss / len(eval_loader)

def train_unet():
  

    model = unet.UNet(inchannels=1, numclasses=1)
    criterion = nn.BCEWithLogitsLoss() # Penser un tester une Dice Loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    transform = transforms.Compose([
        transforms.Resize((512, 512)), # Resizer
        transforms.ToTensor()  # Convertir en tensor
    ])

    train_dataset, eval_dataset = datasets.make_BrainTumorSegDataset(
        r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\images",
        r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\masks",
        transform=transform
    )
    # train_dataset, eval_dataset = datasets.make_BrainTumorSegDataset(
    #     r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\forest_data\images",
    #     r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\forest_data\masks",
    #     transform=transform
    # )
    train_loader = DataLoader(train_dataset, batch_size = 8, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size = 8, shuffle=True)

    num_epochs = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_eval_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

            if masks.dtype != torch.float32:
                masks = masks.float()

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        eval_loss = evaluate(model, eval_loader, criterion, device)

        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {train_loss}, Eval Loss: {eval_loss}')

        # Enregistrer le modèle si la performance sur l'ensemble de validation s'améliore
        if eval_loss < best_eval_loss:
            best_eval_loss = eval_loss
            torch.save(model.state_dict(), 'best_model.pth')
            print(f'Model saved at epoch {epoch+1}')

import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
from models.unet import UNet_2 as UNET2
from utils import (
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16
NUM_EPOCHS = 10
NUM_WORKERS = 2
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\train_images"
TRAIN_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\train_masks"
VAL_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\val_images"
VAL_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\val_masks"

def train_unet_2(loader, model, optimizer, loss_fct, scaler):
    loop = tqdm(loader)
    
    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().unsqueeze(1).to(device=DEVICE)
        
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

    

def main():
    train_transform = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Rotate(limit=35, p=1.0),
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

    val_transforms = A.Compose(
        [
            A.Resize(height=IMAGE_HEIGHT, width=IMAGE_WIDTH),
            A.Normalize(
                mean=[0.0, 0.0, 0.0],
                std=[1.0, 1.0, 1.0],
                max_pixel_value=255.0,
            ),
            ToTensorV2(),
        ]
    )

    model = UNET2(in_channels=1, out_channels=1)
    model.to(DEVICE)
    loss_fct = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    
    train_loader, val_loader = get_loaders(
        TRAIN_IMG_DIR,
        TRAIN_MASKS_DIR,
        VAL_IMG_DIR,
        VAL_MASKS_DIR,
        BATCH_SIZE,
        train_transform,
        val_transforms
    )

    scaler = torch.amp.GradScaler('cuda')
    for epoch in range(NUM_EPOCHS):
        train_unet_2(train_loader, model, optimizer, loss_fct, scaler)

        # save model
        checkpoint = {
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        save_checkpoint(checkpoint)

        if LOAD_MODEL:
            load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)


        check_accuracy(val_loader, model, device=DEVICE)
        scaler = torch.amp.GradScaler('cuda')

        # print some examples to a folder
        save_predictions_as_imgs(
            val_loader, model, folder="saved_images/", device=DEVICE
        )

if __name__ == "__main__":
    main()

# train_unet()


