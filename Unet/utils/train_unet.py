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
        transforms.Resize((256, 256)), # Resizer
        transforms.ToTensor()  # Convertir en tensor
    ])

    train_dataset, eval_dataset = datasets.make_BrainTumorSegDataset(
        r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\images",
        r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\masks",
        transform=transform
    )
    train_loader = DataLoader(train_dataset, batch_size = 4, shuffle=True)
    eval_loader = DataLoader(eval_dataset, batch_size = 4, shuffle=True)

    num_epochs = 3
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    best_eval_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)

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

train_unet()
