import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# Chemins des dossiers et du fichier CSV
imgs_dir = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\images"
masks_dir = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\masks"
csv_path = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\images_classes.csv"
output_folder = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\output"

# Lire le fichier CSV
df = pd.read_csv(csv_path)

# Obtenir les clusters uniques
unique_clusters = df['cluster'].unique()

# Créer les dossiers de sortie pour chaque cluster
for cluster in unique_clusters:
    train_imgs_dir = os.path.join(output_folder, f"images_cluster_{cluster}", "train")
    train_masks_dir = os.path.join(output_folder, f"masks_cluster_{cluster}", "train")
    eval_imgs_dir = os.path.join(output_folder, f"images_cluster_{cluster}", "eval")
    eval_masks_dir = os.path.join(output_folder, f"masks_cluster_{cluster}", "eval")

    os.makedirs(train_imgs_dir, exist_ok=True)
    os.makedirs(train_masks_dir, exist_ok=True)
    os.makedirs(eval_imgs_dir, exist_ok=True)
    os.makedirs(eval_masks_dir, exist_ok=True)

# Diviser les données en train et eval pour chaque cluster
for cluster in unique_clusters:
    cluster_df = df[df['cluster'] == cluster]
    train_df, eval_df = train_test_split(cluster_df, test_size=0.2, random_state=42)

    # Copier les images et les masques dans les dossiers train
    for index, row in train_df.iterrows():
        img_id = row['id']
        img_path = os.path.join(imgs_dir, img_id + ".png")  # Assuming images are in PNG format
        mask_path = os.path.join(masks_dir, img_id + ".png")  # Assuming masks are in PNG format

        dest_img_path = os.path.join(output_folder, f"images_cluster_{cluster}", "train", img_id + ".png")
        dest_mask_path = os.path.join(output_folder, f"masks_cluster_{cluster}", "train", img_id + ".png")

        if os.path.exists(img_path):
            shutil.copy(img_path, dest_img_path)
        if os.path.exists(mask_path):
            shutil.copy(mask_path, dest_mask_path)

    # Copier les images et les masques dans les dossiers eval
    for index, row in eval_df.iterrows():
        img_id = row['id']
        img_path = os.path.join(imgs_dir, img_id + ".png")  # Assuming images are in PNG format
        mask_path = os.path.join(masks_dir, img_id + ".png")  # Assuming masks are in PNG format

        dest_img_path = os.path.join(output_folder, f"images_cluster_{cluster}", "eval", img_id + ".png")
        dest_mask_path = os.path.join(output_folder, f"masks_cluster_{cluster}", "eval", img_id + ".png")

        if os.path.exists(img_path):
            shutil.copy(img_path, dest_img_path)
        if os.path.exists(mask_path):
            shutil.copy(mask_path, dest_mask_path)

print("Images and masks have been copied to the respective cluster folders.")