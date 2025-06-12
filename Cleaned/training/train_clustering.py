############################################
import os
import sys
parent_folder_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_folder_path)
############################################
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import pandas as pd

import preprocessing.preprocessing as preprocessing

def train_k_means(dataset, **params):
    """Train K-Means algorithm

    Args:
        dataset (pandas.DataFrame): dataset
        scaler (scaler): scaler
    """
    scaled_dataset = params["SCALER"].fit_transform(dataset.values)

    kmeans = KMeans(n_clusters=params["N_CLUSTER"], random_state=42)
    kmeans.fit(scaled_dataset)
    labels = kmeans.labels_

    dataset['cluster'] = labels

if __name__ == "__main__":
    params = {
        "IMGS_DIR": r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\images",
        "MASKS_DIR": r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\masks",
        "CSV_PATH": r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\preprocessed\images_classes.csv",
        "SCALER": StandardScaler(),   
        "N_CLUSTER": 3 
    }

    dataset = preprocessing.extract_tumor_features(params['IMGS_DIR'], params["MASKS_DIR"])
    
    ids = dataset['id']
    dataset.drop(columns=['id', 'eccentricity', 'solidity', 'extent', 'perimeter'], inplace=True)
    
    print(dataset)
    
    train_k_means(dataset, **params)

    id_cluster = pd.concat([ids, dataset['cluster']], ignore_index=False, axis=1)

    id_cluster.to_csv(params["CSV_PATH"], index=False)
