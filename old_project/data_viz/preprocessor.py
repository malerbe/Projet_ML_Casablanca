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

def group_flights(dataset_df):
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

def delete_airline_from_dataset(dataset_df, aita_code):
    if aita_code == "NaN":
        return dataset_df[~dataset_df["Airline IATA Code"].isna()]
    elif type(aita_code) == list:
        for code in aita_code:
            dataset_df = dataset_df[dataset_df["Airline IATA Code"] != code]
        return dataset_df
    else:
        return dataset_df[dataset_df["Airline IATA Code"] != aita_code]


def preprocess_dataset(dataset_df, del_nan_airlines=True, drop_unknown_status=True, Unwanted_airlines=[], grouped=True, keep_values_ADTNan_EDT_over_x_minutes=None):
    """
    Preprocess Dataset with options
    """


    final_dataset = dataset_df.copy()

    # Drop useless columns:
    final_dataset.drop(["Airline Name", "Airline ICAO Code", "Departure Airport ICAO", "Arrival Airport ICAO"], inplace=True, axis=1)

    # Convert types:
    convert_date_columns(final_dataset, ["Scheduled Departure Time", "Estimated Departure Time", "Actual Departure Time", "Scheduled Arrival Time", "Estimated Arrival Time"])
    
    
    #make_departure_delay_column(final_dataset)
    #make_estimated_departure_delay_column(final_dataset)


    ####################################################
    ############# Delete Unwanted Airlines #############
    ####################################################
    if del_nan_airlines:
        # Compagnies type army:
        final_dataset = delete_airline_from_dataset(final_dataset, "NaN")

    # Delete other companies from list:    
    final_dataset = delete_airline_from_dataset(final_dataset, Unwanted_airlines)

    ####################################################
    ############### Drop Unknown Status ################
    ####################################################
    if drop_unknown_status:
        final_dataset = final_dataset[final_dataset["Flight Status"] != "unknown"]

    ####################################################
    ################## Group flights ###################
    ####################################################
    if grouped:
        final_dataset = group_flights(final_dataset)

    ####################################################
    ################## Compute Delays ##################
    ####################################################

    #### Cas numéro 1: Actual Departure Time présent:
    # final_dataset_ADT = final_dataset[~final_dataset["Actual Departure Time"].isna()]

    # # Dans ce cas, on calcule le retard de façon précise:
    # final_dataset_ADT["Delay"] = final_dataset_ADT["Actual Departure Time"] - final_dataset_ADT["Scheduled Departure Time"]

    #### Cas numéro 2: Actual Departure Time absent mais Estimated Departure Time présent:
    ### Cas numéro 2.1: On garde tout

    ### Cas numéro 2.2: On vire tous les Delays en dessous de keep_values_ADTNan_EDT_over_x_minutes

    ### Cas numéro 2.3: On considère tout les Delays au dessus / en dessous d'une limite comme étant supérieurs/inférieurs à une durée

    ### Cas numéro 2.4: On ajoute x minutes au delay, ou on ajoute un delay aléatoire, respectant une loi de distribution à déterminer manuellement grâce à FlightRadar

    return final_dataset
    


