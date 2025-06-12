############################################
import os
import sys
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder_path)
############################################

import torch
import pandas as pd
from torch.utils.data import DataLoader
import torch.nn.functional as F

import datasets.datasets as datasets
import models.classification_cnn
import preprocessing.preprocessing as preprocessing
import utils.utils as utils
import models

def get_datasets(imgs_dir, csv_path, **params):
    train_transforms, test_transforms = preprocessing.get_transformations(img_size=params['IMAGE_WIDTH_HEIGHT'][0])

    dataframe = pd.read_csv(csv_path)

    train_dataframe, test_dataframe = utils.split_dataframe(dataframe, train_perc=0.80)

    train_dataset = datasets.TumorClassesDataset(imgs_dir, train_dataframe, train_transforms)
    test_dataset = datasets.TumorClassesDataset(imgs_dir, test_dataframe, test_transforms)

    return train_dataset, test_dataset

import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return(torch.tensor(torch.sum(preds == labels).item()/len(preds)))

def evaluate(model, val_loader, loss_fct, device):
    outputs = []
    for batch in val_loader:
        images = batch[0].to(device=device)
        labels = batch[1].to(device=device)
        out = model(images)
        loss = loss_fct(out, labels)
        acc = accuracy(out, labels)
        outputs.append({'val_loss': loss, 'val_acc': acc})

    batch_losses = [x['val_loss'] for x in outputs]
    epoch_loss = torch.stack(batch_losses).mean()
    batch_accs = [x['val_acc'] for x in outputs]
    epoch_acc = torch.stack(batch_accs).mean()
    return({'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()})

def fit(epochs, lr, model, train_loader, test_loader, device, loss_fct, opt_func = torch.optim.SGD):
    losses = {"epochs": [],
              "train_losses": [],
              "eval_losses": [],
              "train_accs": [],
              "eval_accs": []}
    optimizer = opt_func(model.parameters(), lr)
    scaler = torch.amp.GradScaler()

    for epoch in range(epochs):
        model.train()
        # Training:
        outputs = []
        for batch in train_loader:
            images = batch[0].to(device=device)
            labels = batch[1].to(device=device)

            optimizer.zero_grad()
            # Forward pass with mixed precision
            with torch.amp.autocast(device):
                out = model(images)
                loss = loss_fct(out, labels)

            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            acc = accuracy(out, labels)
            outputs.append({'train_loss': loss, 'train_acc': acc})

        # Validation
        model.eval()
        batch_losses = [x['train_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['train_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()

        result = evaluate(model, test_loader, loss_fct, device)
        losses['epochs'].append(epoch)
        losses['train_losses'].append(epoch_loss.item())
        losses['train_accs'].append(epoch_acc.item())
        losses['eval_losses'].append(result['val_loss'])
        losses['eval_accs'].append(result['val_acc'])
        print(f"Epoch {epoch}: {result}")

    return losses

def save_results(results_dir, params, losses, model):
    # Create the results directory
    os.makedirs(results_dir, exist_ok=True)

    # Save the model
    model_path = os.path.join(results_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)

    # Save the parameters
    params_path = os.path.join(results_dir, 'params.txt')
    with open(params_path, 'w') as f:
        for key, value in params.items():
            f.write(f"{key}: {value}\n")

    # Plot and save the loss and accuracy curves
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(losses['epochs'], losses['train_losses'], label='Train Loss')
    plt.plot(losses['epochs'], losses['eval_losses'], label='Eval Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')

    plt.subplot(1, 2, 2)
    plt.plot(losses['epochs'], losses['train_accs'], label='Train Accuracy')
    plt.plot(losses['epochs'], losses['eval_accs'], label='Eval Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy Curves')

    plot_path = os.path.join(results_dir, 'loss_accuracy_curves.png')
    plt.savefig(plot_path)
    plt.close()

def train_cnn(**params):
    # Initialize parameters:
    imgs_dir = params["IMGS_DIR"]
    csv_path = params["CSV_PATH"]

    batch_size = params["BATCH_SIZE"]
    epochs = params["EPOCHS"]
    loss_fct = params["LOSS_FUNCTION"]

    device = params["DEVICE"]
    print(f"Starting training on device: {device}")

    # 1. Get the train and test dataframes:
    train_dataset, test_dataset = get_datasets(imgs_dir, csv_path, **params)
    
    # 2. Make loaders:
    train_loader, test_loader = DataLoader(train_dataset, batch_size, shuffle=True), DataLoader(test_dataset, batch_size, shuffle=True)

    # 3. Get the model:
    model = models.classification_cnn.ClassificationCNN(shape_in=[1, params["IMAGE_WIDTH_HEIGHT"][0], params["IMAGE_WIDTH_HEIGHT"][1]], num_classes=params["NUM_CLASSES"], configuration=params["CONFIGURATION"])
    model.to(device=device)

    # 4. Train model:
    results = fit(epochs, 0.001, model, train_loader, test_loader, device, loss_fct)

    # 5. Save model, paramters, and figures:
    results_dir = os.path.join(params["RESULTS_DIR"], f"training_results_{params['EPOCHS']}epochs_{params['BATCH_SIZE']}batch")
    save_results(results_dir, params, results, model)
    print(f"Results saved to {results_dir}")



if __name__ == "__main__":
    params = {
        "IMGS_DIR": r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\images",
        "CSV_PATH": r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\images_classes.csv",
        "IMAGE_WIDTH_HEIGHT" : [128, 128],
        "BATCH_SIZE": 32,
        "EPOCHS": 10,
        "DEVICE": "cuda" if torch.cuda.is_available() else "cpu",
        "LOSS_FUNCTION": F.cross_entropy,
        "NUM_CLASSES": 3,
        "CONFIGURATION": [64, 128, 256],
        "RESULTS_DIR": r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\training_results_2"
    }

    train_cnn(**params)

