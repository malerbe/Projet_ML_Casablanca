import torch
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn.functional as F
import torch.nn as nn
from torch.utils.data import Dataset, random_split, DataLoader
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms

# --- Définir le Dataset personnalisé ---
class TumorClassesDataset(Dataset):
    def __init__(self, images_dir, dataframe, transform=None):
        super().__init__()
        self.images_dir = images_dir
        self.dataframe = dataframe
        self.transform = transform

    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, index):
        img_name = str(self.dataframe.iloc[index, 0])
        if not "." in img_name:
            img_path = os.path.join(self.images_dir, img_name + ".png")
        else:
            img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("L")
        label = int(self.dataframe.iloc[index, 1])

        if self.transform is not None:
            image = self.transform(image)
        return image, label

# --- Transformation pour 128x128 ---
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

# --- Charger le dataframe ---
csv_path = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\images_classes.csv"
images_dir = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\images"
df = pd.read_csv(csv_path)

# --- Créer le dataset ---
full_dataset = TumorClassesDataset(images_dir, df, transform=transform)
print("Nombre d'images :", len(full_dataset))

image, label = full_dataset[10]
plt.imshow(image.squeeze(), cmap='gray')
plt.title(f"Label: {label}")
plt.show()
print("Shape d'un tensor image:", image.shape)  # torch.Size([1, 128, 128])

# --- Split ---
train_size = int(0.8 * len(full_dataset))
val_size = len(full_dataset) - train_size
train_data, validation_data = random_split(full_dataset, [train_size, val_size])
print(f"Train size: {len(train_data)}, Validation size: {len(validation_data)}")

# --- Modèle puissant pour 128x128 ---
class TumorModel(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 128 → 64
            nn.Dropout(0.2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 64 → 32
            nn.Dropout(0.3),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),          # 32 → 16
            nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 16 * 16, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )

    def forward(self, xb):
        xb = self.features(xb)
        out = self.classifier(xb)
        return out

    def training_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        return loss

    def validation_step(self, batch):
        images, labels = batch
        out = self(images)
        loss = F.cross_entropy(out, labels)
        acc = accuracy(out, labels)
        return {'val_loss': loss, 'val_acc': acc}

    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}

    def epoch_end(self, epoch, result):
        print(f"Epoch [{epoch+1}], val_loss: {result['val_loss']:.4f}, val_acc: {result['val_acc']:.4f}")

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

def evaluate(model, val_loader):
    outputs = [model.validation_step(batch) for batch in val_loader]
    return model.validation_epoch_end(outputs)

def fit(epochs, lr, model, train_loader, val_loader, opt_func=torch.optim.Adam):
    history = []
    optimizer = opt_func(model.parameters(), lr)
    for epoch in range(epochs):
        print(f"EPOCH {epoch+1} STARTING")
        model.train()
        for batch in train_loader:
            loss = model.training_step(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        model.eval()
        result = evaluate(model, val_loader)
        model.epoch_end(epoch, result)
        history.append(result)
    return history

batch_size = 32
train_loader = DataLoader(train_data, batch_size, shuffle=True)
val_loader = DataLoader(validation_data, batch_size, shuffle=False)

model = TumorModel(num_classes=3)

history1 = fit(epochs=100, lr=0.001, model=model, train_loader=train_loader, val_loader=val_loader)