import os
import sys

############################################
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder_path)
############################################

import albumentations as A
from albumentations.pytorch import ToTensorV2
import glob
from skimage.io import imread
from skimage.measure import label, regionprops
import numpy
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2

def extract_tumor_features(images_path, masks_path):
    """Extracts features from the portion of the images under
    the mask.
    Will be mainly use to make data which will be used to fit
    a clustering algorithm (K-means)

    Args:
        images_path (str): path to the folder containing the images
        masks_path (str): path to the folder containing the masks
    """
    
    # Get paths for individual images:
    images_paths = sorted(glob.glob(images_path + r"\*.png"))
    masks_paths = sorted(glob.glob(masks_path + r"\*.png"))
    
    # Check if the amounts correspond to eachother:
    assert len(images_paths) == len(masks_paths), "The number of images must be the same as the number of masks"
    
    features = {"id": [],
                "mean_intensity": [],
                "std_intensity": [],
                "max_intensity": [],
                "min_intensity": [],
                "median_intensity": [],
                "area": [],
                "perimeter": [],
                "eccentricity": [],
                "solidity": [],
                "extent": []}
    
    for k in range(len(images_paths)):
        # Load image and mask:
        image_path, mask_path = images_paths[k], masks_paths[k]
        image = imread(image_path, as_gray=True)
        mask = imread(mask_path, as_gray=True)

        mask = (mask>0.99999).astype(np.uint8)

        # Extract the pixels corresponding to the tumor:
        tumor_pixels = image[mask == 1]

        if tumor_pixels.size != 0:
            # Compute features:
            mean_intensity = np.mean(tumor_pixels)
            std_intensity = np.std(tumor_pixels)
            max_intensity = np.max(tumor_pixels)
            min_intensity = np.min(tumor_pixels)
            median_intensity = np.median(tumor_pixels)

            labeled_mask = label(mask)
            props = regionprops(labeled_mask)[0]
            area = props.area
            perimeter = props.perimeter
            eccentricity = props.eccentricity
            solidity = props.solidity
            extent = props.extent

            features["id"].append(image_path.split("\\")[-1].split(".")[0])
            features["mean_intensity"].append(mean_intensity)
            features["std_intensity"].append(std_intensity)
            features["max_intensity"].append(max_intensity)
            features["min_intensity"].append(min_intensity)
            features["median_intensity"].append(median_intensity)
            features["area"].append(area)
            features["perimeter"].append(perimeter)
            features["eccentricity"].append(eccentricity)
            features["solidity"].append(solidity)
            features["extent"].append(extent)

    return pd.DataFrame(features)



def get_transformations(config_name="Basic++", img_size=128):
    if config_name == "Basic++":
        """ Most simple configuration:
        Transformations are just a ToTensorV2
        """

        train_transforms = A.Compose([
                A.Resize(height=img_size, width=img_size),
                ToTensorV2(),
        ])

        eval_transforms = A.Compose([
                A.Resize(height=img_size, width=img_size),
                ToTensorV2(),
        ])
        
        return train_transforms, eval_transforms


def preprocess_dataset(images_dir, masks_dir, output_dir="./preprocessed_data", img_size=512,
                       horizontal_flip=False, vertical_flip=False, random_rotation=False,
                       elastic_deformation=False):
    assert os.path.exists(images_dir), f"Dossier des images introuvale: {images_dir}"
    assert os.path.exists(masks_dir), f"Dossier des images introuvale: {masks_dir}"


    # Output folders :
    processed_images_dir = os.path.join(output_dir, "images")
    processed_masks_dir = os.path.join(output_dir, "masks")
    os.makedirs(processed_images_dir, exist_ok=True)
    os.makedirs(processed_masks_dir, exist_ok=True)

    # Tansformations pipeline :
    augmentations = []
    # 1. Resizing:
    augmentations_start = []
    augmentations_start.append(
        A.Resize(img_size, img_size)
    )

    # 2. Horizontal flip :
    if horizontal_flip:
        augmentations.append(A.HorizontalFlip(p=0.75))
    
    # 3. Vertical flip:
    if vertical_flip:
        augmentations.append(A.VerticalFlip(p=0.75))

    # 4. Rotation:
    if random_rotation:
        augmentations.append(A.Rotate(limit=30, p=1))

    # 5. Elastic deformation:
    if elastic_deformation:
        augmentations.append(A.ElasticTransform(alpha=500, sigma=100, p=1))

    # 6. Normalization and transformation to a tensor:
    augmentations_end = []
    augmentations_end.append(A.Normalize(mean=[0.5], std=[0.5]))
    augmentations_end.append(ToTensorV2())

    pipelines = []
    final_aug = augmentations_start + augmentations + augmentations_end
    pipelines.append(A.Compose(final_aug))
    for k in range(4):
        augmentations = [augmentations[-1]] + augmentations[0:-1]
        final_augs = augmentations_start + augmentations + augmentations_end
        pipelines.append(A.Compose(final_augs))

    images_files = sorted(os.listdir(images_dir))
    masks_files = sorted(os.listdir(masks_dir))
    assert len(images_files) == len(masks_files), "Number of images != number of masks"

    for idx, (img_file, mask_file) in tqdm(
        enumerate(zip(images_files, masks_files)),
        total=len(images_files),
        desc="Preprocessing images..."):

        img_path = os.path.join(images_dir, img_file)
        mask_path = os.path.join(masks_dir, mask_file)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

        if img is None or mask is None:
            raise FileNotFoundError(f"{img_path} or {mask_path} not found.")

        ## Save original files:
        # Apply minimum transformations to the original image:
        basic_pipeline = A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.5], std=[0.5]),
            ToTensorV2(),
        ])
        transformed = basic_pipeline(image=img, mask=mask)
        transformed_img = transformed['image'].squeeze().numpy()
        transformed_mask = transformed['mask'].squeeze().numpy()

        # Last transformation to make the .png visible, easy to transform back:
        transformed_img = ((transformed_img + 1)/2 * 255)

        # Save file:
        original_img_path = os.path.join(processed_images_dir, f"original_{idx:03d}.png")
        original_mask_path = os.path.join(processed_masks_dir, f"original_{idx:03d}.png")
        cv2.imwrite(original_img_path, transformed_img)
        cv2.imwrite(original_mask_path, transformed_mask)

        ## Save transformed files:
        ## Save 5 augmented versions:
        for aug_idx, aug_pipeline in enumerate(pipelines):
            transformed = aug_pipeline(image=img, mask=mask)
            transformed_img = transformed['image'].squeeze().numpy()
            transformed_mask = transformed['mask'].squeeze().numpy()
            transformed_img_path = os.path.join(processed_images_dir, f"augmented{aug_idx}_{idx:03d}.png")
            transformed_mask_path = os.path.join(processed_masks_dir, f"augmented{aug_idx}_{idx:03d}.png")
            # Last transformation to make the .png visible, easy to transform back:
            transformed_img = ((transformed_img + 1)/2 * 255)
            cv2.imwrite(transformed_img_path, transformed_img)
            cv2.imwrite(transformed_mask_path, transformed_mask)

    print(f"Preprocessing done and savec in {output_dir}")





if __name__ == "__main__":
    IMGS_DIR= r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\images"
    MASKS_DIR= r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\masks"
    OUTPUT_DIR= r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed"


    preprocess_dataset(IMGS_DIR, MASKS_DIR, OUTPUT_DIR, img_size=512, horizontal_flip=True, vertical_flip=True, random_rotation=True, elastic_deformation=False)





