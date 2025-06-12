from torch.utils.data import Dataset
import torch

from PIL import Image

import os
import numpy as np

############################################################
# Dataset for segmentation tasks:
class TumorSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)


    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        mask_path = os.path.join(self.mask_dir, self.images[index].replace(".jpg", ".png"))
        image  = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255
        mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32) /255

        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations["image"]
            mask = augmentations["mask"]
        
        return image, mask

class MaskClassificationDataset(Dataset):
    def __init__(self, image_dir, model1_mask_dir, model2_mask_dir, model3_mask_dir, original_mask_dir, transform=None):
        self.image_dir = image_dir
        self.model1_mask_dir = model1_mask_dir
        self.model2_mask_dir = model2_mask_dir
        self.model3_mask_dir = model3_mask_dir
        self.original_mask_dir = original_mask_dir
        self.transform = transform
        self.images = os.listdir(image_dir)

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        img_path = os.path.join(self.image_dir, self.images[index])
        model1_mask_path = os.path.join(self.model1_mask_dir, self.images[index].replace(".jpg", ".png"))
        model2_mask_path = os.path.join(self.model2_mask_dir, self.images[index].replace(".jpg", ".png"))
        model3_mask_path = os.path.join(self.model3_mask_dir, self.images[index].replace(".jpg", ".png"))
        original_mask_path = os.path.join(self.original_mask_dir, self.images[index].replace(".jpg", ".png"))
        
        # Load image and masks
        image = np.array(Image.open(img_path).convert("L"), dtype=np.float32) / 255
        model1_mask = np.array(Image.open(model1_mask_path).convert("L"), dtype=np.float32) / 255
        model2_mask = np.array(Image.open(model2_mask_path).convert("L"), dtype=np.float32) / 255
        model3_mask = np.array(Image.open(model3_mask_path).convert("L"), dtype=np.float32) / 255
        original_mask = np.array(Image.open(original_mask_path).convert("L"), dtype=np.float32) / 255

        if self.transform is not None:
            augmentations = self.transform(
                image=image, 
                masks=[model1_mask, model2_mask, model3_mask, original_mask]
            )
            image = augmentations["image"]
            model1_mask, model2_mask, model3_mask, original_mask = augmentations["masks"]
        
        # Convert to torch tensors and add channel dimension
        image = torch.tensor(image)  # [1, H, W]
        model1_mask = torch.tensor(model1_mask).unsqueeze(0)  # [1, H, W]
        model2_mask = torch.tensor(model2_mask).unsqueeze(0)  # [1, H, W]
        model3_mask = torch.tensor(model3_mask).unsqueeze(0)  # [1, H, W]
        original_mask = torch.tensor(original_mask).unsqueeze(0)  # [1, H, W]

        # Concaténer image + 3 masques comme input
        input_tensor = torch.cat([image, model1_mask, model2_mask, model3_mask], dim=0)  # [4, H, W]
        
        return input_tensor, original_mask


############################################################
# Dataset for classification tasks:

class TumorClassesDataset(Dataset):
    def __init__(self, images_dir, dataframe, transform=None):
        """Initialize the dataset

        Args:
            images_dir (str): directory to the images
            dataframe (pandas.DataFrame): Pandas Df with two columns: img_name and class
            transform (albumentations.Compose, optional): Albumentations transformations to apply. Defaults to None.
        """
        super().__init__()

        self.images_dir = images_dir
        self.dataframe = dataframe
        self.transform = transform

        # Get the images names from the specified directory:
        self.images = os.listdir(images_dir)

    def __len__(self):
        return self.dataframe.shape[0]
    
    def __getitem__(self, index):
        if not "." in str(self.dataframe.iloc[index, 0]):
            img_path = os.path.join(self.images_dir, str(self.dataframe.iloc[index, 0]) + ".png")
        else:
            img_path = os.path.join(self.images_dir, str(self.dataframe.iloc[index, 0]))
        image  = np.array(Image.open(img_path).convert("L"), dtype=np.float32)/255
        label = self.dataframe.iloc[index, 1]

        if self.transform is not None:
            augmentations = self.transform(image=image)
            image = augmentations["image"]

        return image, label

if __name__ == "__main__":
    ds = TumorClassesDataset(r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\images", r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\csv\image_sizes.csv")
