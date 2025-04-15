import pandas as pd
import matplotlib.pyplot as plt


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







