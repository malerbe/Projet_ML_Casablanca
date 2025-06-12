import pandas as pd

def split_dataframe(dataframe, train_perc=0.80):
    """Split a pandas dataframe 

    Args:
        train_perc (float, optional): Percentage of the dataset used for training. Defaults to 0.80.
    """
    train_split = dataframe[:int(dataframe.shape[0]*train_perc)]
    test_split = dataframe[int(dataframe.shape[0]*train_perc):]
    
    return train_split, test_split


