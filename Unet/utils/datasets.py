import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
from PIL import Image
import os

class BrainTumorSegDataset(Dataset):
    def __init__(self, img_paths, masks_paths, transform=None):
        self.img_paths = img_paths
        self.masks_paths = masks_paths
        self.transform = transform

    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, index):
        image  = Image.open(self.img_paths[index])
        mask = Image.open(self.masks_paths[index])

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)
        
        return image, mask
    

def make_BrainTumorSegDataset(imgs_path, masks_path, transform=None, train = 0.2):
    imgs_paths, masks_paths = [os.path.join(imgs_path, path) for path in os.listdir(imgs_path)], [os.path.join(masks_path, path) for path in os.listdir(masks_path)]

    assert len(imgs_paths) == len(masks_paths)

    train_imgs_paths, train_masks_paths = imgs_paths[:int(len(imgs_paths)*train)], masks_paths[:int(len(masks_paths)*train)]
    eval_imgs_paths, eval_masks_paths = imgs_paths[int(len(imgs_path)*train):], masks_paths[int(len(masks_path)*train):]
    
    train_dataset = BrainTumorSegDataset(train_imgs_paths, train_masks_paths, transform)
    eval_dataset = BrainTumorSegDataset(eval_imgs_paths, eval_masks_paths, transform)

    return train_dataset, eval_dataset

