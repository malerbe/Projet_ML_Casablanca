import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import os

class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.image_files = sorted(os.listdir(image_dir))
        self.mask_files = sorted(os.listdir(mask_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_files[idx])

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # Convert to grayscale

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask

def check_mask_dimensions(dataset):
    for i, (image, mask) in enumerate(dataset):
        print(f"Image {i+1}: Dimensions = {image.shape}, Data Type = {image.dtype}")
        print(f"Mask {i+1}: Dimensions = {mask.shape}, Data Type = {mask.dtype}")

def main():
    image_dir = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\forest_data\images"
    mask_dir = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\forest_data\masks"

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resizer
        transforms.ToTensor()  # Convertir en tensor
    ])

    dataset = CustomDataset(image_dir, mask_dir, transform=transform)
    check_mask_dimensions(dataset)

if __name__ == "__main__":
    main()