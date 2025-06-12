import os
import glob
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.io import imread
from skimage.measure import label, regionprops
from skimage.morphology import binary_erosion, binary_dilation
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

def analyze_dataset(images_path, masks_path, output_dir="./analysis_results"):
    """
    Analyse complète du dataset d'images médicales avec masques de tumeurs.
    
    Args:
        images_path (str): Chemin vers le dossier contenant les images
        masks_path (str): Chemin vers le dossier contenant les masques
        output_dir (str): Dossier de sortie pour sauvegarder les résultats
    """
    
    # Créer le dossier de sortie
    os.makedirs(output_dir, exist_ok=True)
    
    # Obtenir les chemins des images et masques
    images_paths = sorted(glob.glob(os.path.join(images_path, "*.png")))
    masks_paths = sorted(glob.glob(os.path.join(masks_path, "*.png")))
    
    # Vérifier la correspondance
    assert len(images_paths) == len(masks_paths), "Le nombre d'images doit être égal au nombre de masques"
    
    print(f"Analyse de {len(images_paths)} images...")
    
    # Initialiser les structures de données pour l'analyse
    analysis_data = {
        "id": [],
        "has_tumor": [],
        "tumor_area": [],
        "tumor_pixels": [],
        "image_area": [],
        "tumor_ratio": [],
        "mean_intensity": [],
        "std_intensity": [],
        "max_intensity": [],
        "min_intensity": [],
        "median_intensity": [],
        "perimeter": [],
        "eccentricity": [],
        "solidity": [],
        "extent": [],
        "compactness": [],
        "aspect_ratio": [],
        "num_components": []
    }
    
    # Analyser chaque image
    for k in tqdm(range(len(images_paths)), desc="Analyse des images"):
        image_path, mask_path = images_paths[k], masks_paths[k]
        
        # Charger l'image et le masque
        image = imread(image_path, as_gray=True)
        mask = imread(mask_path, as_gray=True)
        
        # Binariser le masque
        mask = (mask > 0.5).astype(np.uint8)
        
        # ID de l'image
        img_id = os.path.basename(image_path).split(".")[0]
        analysis_data["id"].append(img_id)
        
        # Vérifier la présence d'une tumeur
        has_tumor = np.sum(mask) > 0
        analysis_data["has_tumor"].append(has_tumor)
        
        # Aires
        image_area = image.shape[0] * image.shape[1]
        analysis_data["image_area"].append(image_area)
        
        if has_tumor:
            # Extraire les pixels de tumeur
            tumor_pixels = image[mask == 1]
            tumor_area = np.sum(mask)
            
            analysis_data["tumor_area"].append(tumor_area)
            analysis_data["tumor_pixels"].append(len(tumor_pixels))
            analysis_data["tumor_ratio"].append(tumor_area / image_area)
            
            # Statistiques d'intensité
            analysis_data["mean_intensity"].append(np.mean(tumor_pixels))
            analysis_data["std_intensity"].append(np.std(tumor_pixels))
            analysis_data["max_intensity"].append(np.max(tumor_pixels))
            analysis_data["min_intensity"].append(np.min(tumor_pixels))
            analysis_data["median_intensity"].append(np.median(tumor_pixels))
            
            # Propriétés morphologiques
            labeled_mask = label(mask)
            props = regionprops(labeled_mask)
            
            if len(props) > 0:
                main_region = props[0]  # Plus grande région
                
                analysis_data["perimeter"].append(main_region.perimeter)
                analysis_data["eccentricity"].append(main_region.eccentricity)
                analysis_data["solidity"].append(main_region.solidity)
                analysis_data["extent"].append(main_region.extent)
                
                # Compacité = 4π * aire / périmètre²
                if main_region.perimeter > 0:
                    compactness = (4 * np.pi * main_region.area) / (main_region.perimeter ** 2)
                else:
                    compactness = 0
                analysis_data["compactness"].append(compactness)
                
                # Ratio d'aspect
                bbox = main_region.bbox
                height = bbox[2] - bbox[0]
                width = bbox[3] - bbox[1]
                aspect_ratio = max(height, width) / max(min(height, width), 1)
                analysis_data["aspect_ratio"].append(aspect_ratio)
                
                # Nombre de composantes connexes
                analysis_data["num_components"].append(len(props))
            else:
                # Valeurs par défaut si aucune région détectée
                analysis_data["perimeter"].append(0)
                analysis_data["eccentricity"].append(0)
                analysis_data["solidity"].append(0)
                analysis_data["extent"].append(0)
                analysis_data["compactness"].append(0)
                analysis_data["aspect_ratio"].append(1)
                analysis_data["num_components"].append(0)
        else:
            # Pas de tumeur - utiliser des valeurs par défaut
            analysis_data["tumor_area"].append(0)
            analysis_data["tumor_pixels"].append(0)
            analysis_data["tumor_ratio"].append(0)
            analysis_data["mean_intensity"].append(0)
            analysis_data["std_intensity"].append(0)
            analysis_data["max_intensity"].append(0)
            analysis_data["min_intensity"].append(0)
            analysis_data["median_intensity"].append(0)
            analysis_data["perimeter"].append(0)
            analysis_data["eccentricity"].append(0)
            analysis_data["solidity"].append(0)
            analysis_data["extent"].append(0)
            analysis_data["compactness"].append(0)
            analysis_data["aspect_ratio"].append(1)
            analysis_data["num_components"].append(0)
    
    # Créer DataFrame
    df = pd.DataFrame(analysis_data)
    
    # Filtrer seulement les images avec tumeurs pour les statistiques
    df_with_tumor = df[df['has_tumor'] == True]
    
    print(f"\n🎯 STATISTIQUES GÉNÉRALES :")
    print(f"   • Total d'images : {len(df)}")
    print(f"   • Images avec tumeur : {len(df_with_tumor)} ({len(df_with_tumor)/len(df)*100:.1f}%)")
    print(f"   • Images sans tumeur : {len(df) - len(df_with_tumor)} ({(len(df) - len(df_with_tumor))/len(df)*100:.1f}%)")
    
    # Sélectionner les colonnes numériques importantes pour le tableau de statistiques
    numeric_columns = [
        'tumor_area', 'tumor_pixels', 'image_area', 'tumor_ratio',
        'mean_intensity', 'std_intensity', 'max_intensity', 'min_intensity', 'median_intensity',
        'perimeter', 'eccentricity', 'solidity', 'extent', 'compactness', 'aspect_ratio', 'num_components'
    ]
    
    # Créer le DataFrame pour les statistiques (seulement images avec tumeur)
    stats_df = df_with_tumor[numeric_columns]
    
    print(f"\n📋 TABLEAU DE STATISTIQUES (Images avec tumeur seulement) :")
    print("="*100)
    
    # Afficher le tableau de statistiques
    summary_stats = stats_df.describe()
    print(summary_stats)
    
    # Sauvegarder le tableau au format CSV
    csv_path = os.path.join(output_dir, "statistics_summary.csv")
    summary_stats.to_csv(csv_path)
    print(f"\n💾 Tableau sauvegardé en CSV : {csv_path}")
    
    # Générer le code LaTeX pour le tableau
    latex_code = generate_latex_table(summary_stats)
    
    # Sauvegarder le code LaTeX
    latex_path = os.path.join(output_dir, "statistics_table.tex")
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_code)
    print(f"📝 Code LaTeX sauvegardé : {latex_path}")
    
    # Afficher le code LaTeX
    print(f"\n📊 CODE LATEX POUR LE TABLEAU :")
    print("="*100)
    print(latex_code)
    print("="*100)
    
    # Sauvegarder le DataFrame complet
    df.to_csv(os.path.join(output_dir, "dataset_analysis_complete.csv"), index=False)
    
    return df

def generate_latex_table(summary_stats):
    """
    Génère le code LaTeX pour un tableau de statistiques.
    """
    
    latex_code = """
\\begin{table}[h]
\\centering
\\caption{Statistiques descriptives du dataset (images avec tumeurs)}
\\label{tab:dataset_stats}
\\resizebox{\\textwidth}{!}{%
\\begin{tabular}{|l|""" + "c|" * len(summary_stats.columns) + """}
\\hline
\\textbf{Statistique}"""
    
    # En-têtes des colonnes
    for col in summary_stats.columns:
        latex_code += f" & \\textbf{{{col.replace('_', '\\_')}}}"
    
    latex_code += " \\\\\n\\hline\n"
    
    # Mapper les index français
    index_mapping = {
        'count': 'Nombre',
        'mean': 'Moyenne',
        'std': 'Écart-type',
        'min': 'Minimum',
        '25%': 'Q1 (25\\%)',
        '50%': 'Médiane (50\\%)',
        '75%': 'Q3 (75\\%)',
        'max': 'Maximum'
    }
    
    # Données du tableau
    for idx in summary_stats.index:
        french_idx = index_mapping.get(idx, idx)
        latex_code += f"{french_idx}"
        
        for col in summary_stats.columns:
            value = summary_stats.loc[idx, col]
            
            # Formatage selon le type de valeur
            if idx == 'count':
                formatted_value = f"{int(value)}"
            elif col in ['tumor_area', 'tumor_pixels', 'image_area', 'perimeter']:
                formatted_value = f"{value:.0f}"
            elif col in ['tumor_ratio', 'eccentricity', 'solidity', 'extent', 'compactness']:
                formatted_value = f"{value:.3f}"
            elif 'intensity' in col:
                formatted_value = f"{value:.2f}"
            elif col == 'aspect_ratio':
                formatted_value = f"{value:.2f}"
            elif col == 'num_components':
                formatted_value = f"{value:.1f}"
            else:
                formatted_value = f"{value:.2f}"
            
            latex_code += f" & {formatted_value}"
        
        latex_code += " \\\\\n"
    
    latex_code += """\\hline
\\end{tabular}%
}
\\end{table}

% Légende des variables :
% tumor_area : Aire de la tumeur (pixels)
% tumor_pixels : Nombre de pixels de tumeur
% image_area : Aire totale de l'image (pixels)
% tumor_ratio : Ratio tumeur/image
% mean_intensity : Intensité moyenne
% std_intensity : Écart-type d'intensité
% max_intensity : Intensité maximale
% min_intensity : Intensité minimale
% median_intensity : Intensité médiane
% perimeter : Périmètre de la tumeur
% eccentricity : Excentricité (0=cercle, 1=ellipse allongée)
% solidity : Solidité (compacité)
% extent : Ratio aire/bounding_box
% compactness : Compacité (4π×aire/périmètre²)
% aspect_ratio : Ratio hauteur/largeur
% num_components : Nombre de composantes connexes
"""
    
    return latex_code

def quick_dataset_overview(images_path, masks_path):
    """Aperçu rapide du dataset"""
    images_paths = sorted(glob.glob(os.path.join(images_path, "*.png")))
    masks_paths = sorted(glob.glob(os.path.join(masks_path, "*.png")))
    
    print(f"🔍 APERÇU RAPIDE DU DATASET:")
    print(f"   • Images trouvées : {len(images_paths)}")
    print(f"   • Masques trouvés : {len(masks_paths)}")
    
    if len(images_paths) != len(masks_paths):
        print("⚠️  ATTENTION: Nombre d'images ≠ nombre de masques!")
        return
    
    # Vérifier quelques images pour la présence de tumeurs
    has_tumor_count = 0
    sample_size = min(50, len(masks_paths))
    
    for i in range(sample_size):
        mask = imread(masks_paths[i], as_gray=True)
        mask = (mask > 0.5).astype(np.uint8)
        if np.sum(mask) > 0:
            has_tumor_count += 1
    
    tumor_percentage = (has_tumor_count / sample_size) * 100
    print(f"   • Échantillon de {sample_size} images: {tumor_percentage:.1f}% avec tumeurs")

if __name__ == "__main__":
    # Chemins vers vos données
    IMGS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\images"
    MASKS_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\masks"
    OUTPUT_DIR = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\analysis"
    
    # Aperçu rapide
    quick_dataset_overview(IMGS_DIR, MASKS_DIR)
    
    # Analyse complète
    df_analysis = analyze_dataset(IMGS_DIR, MASKS_DIR, OUTPUT_DIR)
    
    print(f"\n✅ Analyse terminée ! Fichiers générés dans {OUTPUT_DIR}")
