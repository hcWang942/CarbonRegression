import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import norm, rankdata, mstats

def clean_data(df):
    """清理数据：去除首行，替换'-'为NaN，转换为数值型，用均值填充缺失值"""
    df_clean = df.iloc[1:].replace('-', np.nan)
    df_clean = df_clean.apply(pd.to_numeric, errors='coerce')
    means = df_clean.mean()
    return df_clean.fillna(means)

def inverse_normal_transform(series):
    """Rank-inverse Transformation"""
    ranks = rankdata(series)
    transformed = norm.ppf((ranks - 0.5) / len(ranks))
    return transformed

def transform_data(df):
    """Main data transformation function
    df: original dataframe
    """
    df_clean = clean_data(df)
    df_rank = df_clean.apply(inverse_normal_transform)
    return df_rank