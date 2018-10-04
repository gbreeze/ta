# -*- coding: utf-8 -*-
import math
import pandas as pd

from sys import float_info as sflt


def dropna(df):
    """Drop rows with 'Nan' values"""
    df = df[df < math.exp(709)] # big number
    df = df[df != 0.0]
    df = df.dropna()
    return df


def ema_depreciated(series:pd.Series, periods:int):
    """Modified EMA with an SMA
    
    Rolled into ema when kwargs['presma'] = True
    """
    series = verify_series(series)
    sma = series.rolling(window=periods, min_periods=periods).mean()[:periods]
    rest = series[periods:]
    return pd.concat([sma, rest]).ewm(span=periods, adjust=False).mean()


def get_drift(x:int):
    """Returns an int if not zero, otherwise defaults to one."""
    return int(x) if x and x != 0 else 1


def get_offset(x:int):
    """Returns an int, otherwise defaults to zero."""
    return int(x) if x else 0


def signed_series(series:pd.Series, initial:int = None):
    """Returns a Signed Series with or without an initial value"""
    series = verify_series(series)
    sign = series.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = initial
    return sign


def verify_series(series:pd.Series):
    """Raises error if not a Pandas Series otherwise returns the Series"""
    if not isinstance(series, pd.Series):
        raise AttributeError(f"{type(series)} is not a Pandas Series")
    return series


def zero(x):
    return 0 if -sflt.epsilon < x and x < sflt.epsilon else x