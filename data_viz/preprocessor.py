import pandas as pd
import matplotlib.pyplot as plt
from tqdm import *

def make_df(filename):
    """
    Inputs:
    filename (str): path to dataset in .csv format

    Returns:
    (pd.Dataframe): Dataframe containing the full dataset
    """

    df = pd.read_csv(filename)
    return df

def convert_date_column(dataset_df, column_name):
    dataset_df[column_name] = pd.to_datetime(dataset_df[column_name])

def convert_date_columns(dataset_df, columns_names):
    for column_name in columns_names:
        convert_date_column(dataset_df, column_name)

def make_departure_delay_column(dataset_df):
    dataset_df["Departure Delay"] = (dataset_df["Actual Departure Time"] - dataset_df["Scheduled Departure Time"]).dt.total_seconds() / 60

def make_estimated_departure_delay_column(dataset_df):
    dataset_df["Estimated Departure Delay"] = (dataset_df["Estimated Departure Time"] - dataset_df["Scheduled Departure Time"]).dt.total_seconds() / 60

def group_flights_optimized(dataset_df):
    # Initialisation d'une liste pour stocker les nouvelles lignes
    rows = []
    
    # Grouper par colonnes hiérarchiques : "Scheduled Departure Time", "Departure Gate", "Arrival Airport IATA"
    grouped = dataset_df.groupby(["Scheduled Departure Time", "Departure Gate", "Arrival Airport IATA"])
    
    # Boucler sur chaque groupe
    for (sch_dt, dep_gate, arr_airport_iata), group in tqdm(grouped, desc="Processing Groups"):
        # Créer une nouvelle ligne pour chaque combinaison unique
        new_row = {
            "Scheduled Departure Time": sch_dt,
            "Departure Gate": dep_gate,
            "Arrival Airport IATA": arr_airport_iata,
            "Flight Number": list(group["Flight Number"].unique()),  # Liste unique des numéros de vol
            "Airline IATA Code": list(group["Airline IATA Code"].unique()),  # Liste unique des compagnies
        }

        # Pour toutes les autres colonnes, appliquer des règles spécifiques
        for column in dataset_df.columns:
            if column not in new_row:
                # Vérifier l'unicité des valeurs pour la colonne
                unique_values = group[column].unique()
                if len(unique_values) == 1:  # Si toutes les valeurs sont identiques
                    new_row[column] = unique_values[0]  # Prendre cette valeur unique
                else:
                    new_row[column] = None  # Marquer comme incohérent si plusieurs valeurs
        
        # Ajouter la ligne au stockage temporaire
        rows.append(new_row)
    
    # Création d'un DataFrame final à partir des lignes concaténées
    final_df = pd.DataFrame(rows)
    
    return final_df