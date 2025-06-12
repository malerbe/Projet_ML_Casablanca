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
import train_segmentation


# ## Entrainement 1: 
# TRAIN_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\images_cluster_0\train"
# TRAIN_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\masks_cluster_0\train"
# VAL_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\images_cluster_0\eval"
# VAL_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\masks_cluster_0\eval"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dice_loss = DiceLoss()
# cross_entropy_loss = nn.CrossEntropyLoss()
# iou_loss = IoULoss()
# focal_loss = FocalLoss()
# CExDL05 = CExDL(alpha=0.5)
# CExDL025 = CExDL(alpha=0.25)
# CExDL075 = CExDL(alpha=0.75)
# DLxFL05 = DLxFL(alpha=0.5)
# DLxFL025 = DLxFL(alpha=0.25)
# DLxFL075 = DLxFL(alpha=0.75)


# params = {
#     "IMAGE_WIDTH_HEIGHT" : [128, 128],
#     "ROTATE_LIMIT": 35,
#     "BATCH_NORM": True,
#     "TRAIN_IMGS_DIR": TRAIN_IMG_DIR,
#     "TRAIN_MASKS_DIR": TRAIN_MASKS_DIR,
#     "EVAL_IMGS_DIR": VAL_IMG_DIR,
#     "EVAL_MASKS_DIR": VAL_MASKS_DIR,
#     "LEARNING_RATE": 1e-4,
#     "DEVICE": DEVICE,
#     "BATCH_SIZE": 16,
#     "EPOCHS": 30,
#     "LOAD_MODEL": False,
#     "LOSS_FUNCTION": DLxFL075,
#     "KERNEL_POOL_SIZE": 2,
#     "CONV_KERNEL_SIZE": 2,
#     "DEEP_SUPERVISION": False,
#     "SCHEDULER": "step_lr",
#     "UNET_FEATURES": [64, 128, 256, 512, 1024],
#     "BATCH_SUPERVISION": False,
#     "SCALER": torch.amp.GradScaler(),
#     "SAVENAME": "CLUSTER0"
# }

# model = unets.NestedUNet(in_channels=1, out_channels=1, nb_filter=params["UNET_FEATURES"], deep_supervision=params["DEEP_SUPERVISION"])  

# params["MODEL"] = model    
# params["OPTIMIZER"] = optim.Adam(model.parameters(), lr=params["LEARNING_RATE"])

# train_segmentation.train_unet(**params)

# ## Entrainement 2: 
# TRAIN_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\images_cluster_1\train"
# TRAIN_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\masks_cluster_1\train"
# VAL_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\images_cluster_1\eval"
# VAL_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\masks_cluster_1\eval"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dice_loss = DiceLoss()
# cross_entropy_loss = nn.CrossEntropyLoss()
# iou_loss = IoULoss()
# focal_loss = FocalLoss()
# CExDL05 = CExDL(alpha=0.5)
# CExDL025 = CExDL(alpha=0.25)
# CExDL075 = CExDL(alpha=0.75)
# DLxFL05 = DLxFL(alpha=0.5)
# DLxFL025 = DLxFL(alpha=0.25)
# DLxFL075 = DLxFL(alpha=0.75)


# params = {
#     "IMAGE_WIDTH_HEIGHT" : [128, 128],
#     "ROTATE_LIMIT": 35,
#     "BATCH_NORM": True,
#     "TRAIN_IMGS_DIR": TRAIN_IMG_DIR,
#     "TRAIN_MASKS_DIR": TRAIN_MASKS_DIR,
#     "EVAL_IMGS_DIR": VAL_IMG_DIR,
#     "EVAL_MASKS_DIR": VAL_MASKS_DIR,
#     "LEARNING_RATE": 1e-4,
#     "DEVICE": DEVICE,
#     "BATCH_SIZE": 16,
#     "EPOCHS": 100,
#     "LOAD_MODEL": False,
#     "LOSS_FUNCTION": DLxFL075,
#     "KERNEL_POOL_SIZE": 2,
#     "CONV_KERNEL_SIZE": 2,
#     "DEEP_SUPERVISION": False,
#     "SCHEDULER": "step_lr",
#     "UNET_FEATURES": [64, 128, 256, 512, 1024],
#     "BATCH_SUPERVISION": False,
#     "SCALER": torch.amp.GradScaler(),
#     "SAVENAME": "CLUSTER1"
# }

# model = unets.NestedUNet(in_channels=1, out_channels=1, nb_filter=params["UNET_FEATURES"], deep_supervision=params["DEEP_SUPERVISION"])  

# params["MODEL"] = model    
# params["OPTIMIZER"] = optim.Adam(model.parameters(), lr=params["LEARNING_RATE"])

# train_segmentation.train_unet(**params)

# ## Entrainement 3: 
# TRAIN_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\images_cluster_2\train"
# TRAIN_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\masks_cluster_2\train"
# VAL_IMG_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\images_cluster_2\eval"
# VAL_MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output\masks_cluster_2\eval"
# DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# dice_loss = DiceLoss()
# cross_entropy_loss = nn.CrossEntropyLoss()
# iou_loss = IoULoss()
# focal_loss = FocalLoss()
# CExDL05 = CExDL(alpha=0.5)
# CExDL025 = CExDL(alpha=0.25)
# CExDL075 = CExDL(alpha=0.75)
# DLxFL05 = DLxFL(alpha=0.5)
# DLxFL025 = DLxFL(alpha=0.25)
# DLxFL075 = DLxFL(alpha=0.75)


# params = {
#     "IMAGE_WIDTH_HEIGHT" : [128, 128],
#     "ROTATE_LIMIT": 35,
#     "BATCH_NORM": True,
#     "TRAIN_IMGS_DIR": TRAIN_IMG_DIR,
#     "TRAIN_MASKS_DIR": TRAIN_MASKS_DIR,
#     "EVAL_IMGS_DIR": VAL_IMG_DIR,
#     "EVAL_MASKS_DIR": VAL_MASKS_DIR,
#     "LEARNING_RATE": 1e-4,
#     "DEVICE": DEVICE,
#     "BATCH_SIZE": 16,
#     "EPOCHS": 100,
#     "LOAD_MODEL": False,
#     "LOSS_FUNCTION": DLxFL075,
#     "KERNEL_POOL_SIZE": 2,
#     "CONV_KERNEL_SIZE": 2,
#     "DEEP_SUPERVISION": False,
#     "SCHEDULER": "step_lr",
#     "UNET_FEATURES": [64, 128, 256, 512, 1024],
#     "BATCH_SUPERVISION": False,
#     "SCALER": torch.amp.GradScaler(),
#     "SAVENAME": "CLUSTER2"
# }

# model = unets.NestedUNet(in_channels=1, out_channels=1, nb_filter=params["UNET_FEATURES"], deep_supervision=params["DEEP_SUPERVISION"])  

# params["MODEL"] = model    
# params["OPTIMIZER"] = optim.Adam(model.parameters(), lr=params["LEARNING_RATE"])

# train_segmentation.train_unet(**params)

