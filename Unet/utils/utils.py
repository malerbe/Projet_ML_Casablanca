import torch
import torchvision
from datasets import BrainTumorSegDataset_2, TumorSizeDataset
from torch.utils.data import DataLoader

def save_checkpoint(state, filename="mycheckpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint, model):
    print("=> Loading Checkpoint")
    model.load_state_dict(checkpoint["state_dict"])


def get_loaders(
    train_dir,
    train_maskdir,
    val_dir,
    val_maskdir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    train_size = 1.0,
):
    train_ds = BrainTumorSegDataset_2(
        image_dir=train_dir,
        mask_dir=train_maskdir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = BrainTumorSegDataset_2(
        image_dir=val_dir,
        mask_dir=val_maskdir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader

def check_accuracy(loader, model, device="cuda"):
    num_correct = 0
    num_pixels = 0
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    print(
        f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}"
    )
    print(f"Dice score: {dice_score/len(loader)}")
    model.train()


def check_accuracy_2(loader, model, device="cuda"):
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
            y = y.to(device).unsqueeze(1)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

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

    print(f"Got {num_correct}/{num_pixels} with acc {accuracy:.2f}%")
    print(f"Dice score: {dice_score:.4f}")
    print(f"Overlap percentage: {overlap_percentage:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    model.train()

def check_accuracy_classification(loader, model, device="cuda"):
    num_correct = 0
    num_samples = 0
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    model.eval()

    with torch.no_grad():
        for x, y, _, _ in loader:
            x = x.to(device)
            y = y.to(device)
            preds = model(x).view(-1)
            preds = (preds > 0.5).float()


            print(preds)
            
            # Calculer le nombre de prédictions correctes
            num_correct += (preds == y).sum().item()
            num_samples += y.size(0)

            # Calculer les vrais positifs, faux positifs et faux négatifs
            true_positives += (preds * y).sum().item()
            false_positives += (preds * (1 - y)).sum().item()
            false_negatives += ((1 - preds) * y).sum().item()

    accuracy = num_correct / num_samples * 100
    precision = true_positives / (true_positives + false_positives + 1e-8)
    recall = true_positives / (true_positives + false_negatives + 1e-8)
    f1_score = 2 * (precision * recall) / (precision + recall + 1e-8)

    print(f"Got {num_correct}/{num_samples} with acc {accuracy:.2f}%")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1_score:.4f}")

    model.train()

    

def save_predictions_as_imgs(loader, model, folder="saved_images/", device="cuda"):
    model.eval()
    for idx, (x, y) in enumerate(loader):
        x = x.to(device=device)
        y = y.to(device=device)

        with torch.no_grad():
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()

        # Convertir l'image d'origine en noir et blanc
        x_bw = x.squeeze(1).cpu()  # Supprimer le canal unique

        # Superposer les masques sur l'image d'origine en noir et blanc
        combined_image = x_bw * 0.5 + y.cpu() * 0.5 + preds.squeeze(1).cpu() * 0.5

        # Sauvegarder l'image finale
        torchvision.utils.save_image(combined_image.unsqueeze(1), f"{folder}/combined_{idx}.png")

    model.train()

def get_loaders_for_classification(
    train_dir,
    val_dir,
    csv_dir,
    batch_size,
    train_transform,
    val_transform,
    num_workers=4,
    pin_memory=True,
    train_size = 1.0):

    train_ds = TumorSizeDataset(
        csv_file = csv_dir,
        image_dir=train_dir,
        transform=train_transform,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=True,
    )

    val_ds = TumorSizeDataset(
        csv_file = csv_dir,
        image_dir=val_dir,
        transform=val_transform,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        shuffle=False,
    )

    return train_loader, val_loader