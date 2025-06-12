import pandas as pd

dataset = pd.read_csv(r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\csv\image_sizes.csv")


dataset["size"] = dataset["size"].apply(lambda x: 0 if x < 6000 else 1)
dataset.to_csv(r"C:\Users\223120964.HCAD\OneDrive - GEHealthCare\Desktop\Projet_ML_Casablanca\Cleaned\data\csv\image_sizes_classes.csv", index=False)
