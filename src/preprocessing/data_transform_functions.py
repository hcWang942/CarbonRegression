import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, rankdata, mstats

def clean_data(df):
    """Clean the data: 
        remove the first row, replace '-' with NaN, 
        convert to numeric,
        fill missing values with the mean.
    """
    df_clean = df.iloc[1:].replace('-', np.nan)
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
    means = df_clean.mean()
    return df_clean.fillna(means)

def inverse_normal_transform(series):
    """Perform Rank-based Inverse Normal Transformation (RINT) on the data."""
    ranks = rankdata(series)
    transformed = norm.ppf((ranks - 0.5) / len(ranks))
    return transformed

def transform_data(df):
    """Main function to transform the data."""
    df_clean = clean_data(df)
    df_rank = df_clean.apply(inverse_normal_transform)
    return df_rank