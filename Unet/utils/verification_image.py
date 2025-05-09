from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

# Charger le masque
mask_path = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\train_masks\1.png"
mask = Image.open(mask_path)

# Convertir l'image en tableau NumPy
mask_array = np.array(mask)

# Afficher les valeurs de pixels
print(mask_array)

# Afficher l'image
plt.imshow(mask_array, cmap='gray')
plt.show()

from PIL import Image
import numpy as np

def compare_png_pixel_values(image_path1, image_path2):
    # Ouvrir les images
    image1 = Image.open(image_path1)
    image2 = Image.open(image_path2)

    # Convertir les images en tableaux NumPy
    image1_array = np.array(image1)
    image2_array = np.array(image2)

    # Calculer les valeurs minimales et maximales des pixels
    min_value1 = np.min(image1_array)
    max_value1 = np.max(image1_array)
    min_value2 = np.min(image2_array)
    max_value2 = np.max(image2_array)

    # Afficher les résultats
    print(f"Image 1 - Min value: {min_value1}, Max value: {max_value1}")
    print(f"Image 2 - Min value: {min_value2}, Max value: {max_value2}")

    # Comparer les valeurs minimales et maximales
    if min_value1 == min_value2 and max_value1 == max_value2:
        print("Les valeurs minimales et maximales des pixels sont identiques pour les deux images.")
    else:
        print("Les valeurs minimales et maximales des pixels sont différentes pour les deux images.")

# Exemple d'utilisation
mask_path_1 = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\data\train_masks\1.png"
mask_path_2 = r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Unet\car_data\masks\0cdf5b5d0ce1_01.png"
compare_png_pixel_values(mask_path_1, mask_path_2)