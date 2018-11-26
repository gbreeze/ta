# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

from functools import reduce
from operator import mul
from sys import float_info as sflt



def combination(n:int, r:int):
    """https://stackoverflow.com/questions/4941753/is-there-a-math-ncr-function-in-python"""
    if r < 0:
        return None
    r = min(n, n - r)
    if r == 0:
        return 1

    numerator   = reduce(mul, range(n, n - r, -1), 1)
    denominator = reduce(mul, range(1, r + 1), 1)
    return numerator // denominator


def dropna(df:pd.DataFrame):
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


def multichoose(n:int, r:int):
    """https://en.wikipedia.org/wiki/Binomial_coefficient"""
    return combination(n + r - 1, r)


def pascals_triangle(n:int):
    """Pascal's Triangle

    Returns a numpy array of the nth row of Pascal's Triangle.
    """
    if n < 0: return None

    # Calculation
    triangle = np.array([combination(n, i) for i in range(0, n + 1)])

    # Variations and Properties
    max_ = np.max(triangle)
    inverted = max_ - triangle
    triangle_sum = np.sum(triangle)
    triangle_avg = np.average(triangle)

    weighted = triangle / triangle_sum
    inv_weighted = inverted / triangle_sum

    return triangle, triangle_sum, triangle_avg, inverted, weighted, inv_weighted, triangle_avg


def signed_series(series:pd.Series, initial:int = None):
    """Returns a Signed Series with or without an initial value"""
    series = verify_series(series)
    sign = series.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = initial
    return sign


def verify_series(series:pd.Series):
    """If a Pandas Series return it."""
    if series is not None and isinstance(series, pd.core.series.Series):
        return series


def zero(x):
    """If the value is close to zero, then return zero.  Otherwise return the value."""
    return 0 if -sflt.epsilon < x and x < sflt.epsilon else x