# -*- coding: utf-8 -*-
import math
import pandas as pd

def verify_series(series:pd.Series):
    if not isinstance(series, pd.Series):
        raise AttributeError(f"{type(series)} is not a Pandas Series")
    return series

def dropna(df):
    """Drop rows with "Nans" values
    """
    df = df[df < math.exp(709)] # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df

def ema(series, periods):
    sma = series.rolling(window=periods, min_periods=periods).mean()[:periods]
    rest = series[periods:]
    return pd.concat([sma, rest]).ewm(span=periods, adjust=False).mean()
