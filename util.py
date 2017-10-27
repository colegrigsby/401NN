import numpy as np
import pandas as pd

def get_split_cols(df):
    numerics = []
    categorical = []

    for col in df:
        if((df[col].dtype == np.float64 or df[col].dtype == np.int64) and col != 'Unnamed: 0'):
            numerics.append(col)

        else:
            categorical.append(col)
            
    return numerics, categorical
    


def get_split_frame(df):
    '''
        Returns numeric, categorical dfs as a tuple
    '''
    numerics, categorical = get_split_cols(df)

    categorical_df = df[categorical]
    numeric_df = df[numerics]
    
    return numeric_df, categorical_df