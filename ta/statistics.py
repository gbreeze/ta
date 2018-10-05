# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .overlap import ema, sma, vwma
from .utils import get_offset, verify_series


def kurtosis(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Kurtosis
    
    Use help(df.ta.kurtosis) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 30
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    kurtosis = close.rolling(length, min_periods=min_periods).kurt()

    # Offset
    kurtosis = kurtosis.shift(offset)

    # Name & Category
    kurtosis.name = f"KURT_{length}"
    kurtosis.category = 'statistics'

    return kurtosis


def mcv(vwap:pd.Series, volume:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Moving Covariance
    
    Use help(df.ta.mcv) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    vwap = verify_series(vwap)
    volume = verify_series(volume)
    length = int(length) if length and length > 0 else 30
    offset = get_offset(offset)

    # Calculate Result
    mean = vwma(close=vwap, volume=volume, length=length)
    std = stdev(close=vwap, length=length)
    mcv = 100 * std / mean

    # Offset
    mcv = mcv.shift(offset)

    # Name & Category
    mcv.name = f"MCV_{length}"
    mcv.category = 'statistics'

    return mcv


def quantile(close:pd.Series, length=None, q=None, offset=None, **kwargs):
    """Indicator: Quantile
    
    Use help(df.ta.quantile) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 30
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    q = float(q) if q and q > 0 and q < 1 else 0.5
    offset = get_offset(offset)

    # Calculate Result
    quantile = close.rolling(length, min_periods=min_periods).quantile(q)

    # Offset
    quantile = quantile.shift(offset)

    # Name & Category
    quantile.name = f"QTL_{length}_{q}"
    quantile.category = 'statistics'

    return quantile


def skew(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Skew
    
    Use help(df.ta.skew) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 30
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    skew = close.rolling(length, min_periods=min_periods).skew()

    # Offset
    skew = skew.shift(offset)

    # Name & Category
    skew.name = f"SKEW_{length}"
    skew.category = 'statistics'

    return skew


def stdev(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Standard Deviation
    
    Use help(df.ta.stdev) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 30
    # min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    stdev = variance(close=close, length=length).apply(np.sqrt)

    # Offset
    stdev = stdev.shift(offset)

    # Name & Category
    stdev.name = f"STDEV_{length}"
    stdev.category = 'statistics'

    return stdev


def variance(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Variance
    
    Use help(df.ta.variance) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 1 else 30
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    variance = close.rolling(length, min_periods=min_periods).var()

    # Offset
    variance = variance.shift(offset)

    # Name & Category
    variance.name = f"VAR_{length}"
    variance.category = 'statistics'

    return variance


def zscore(close:pd.Series, length=None, std=None, offset=None, **kwargs):
    """Indicator: Z Score
    
    Use help(df.ta.zscore) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 1 else 30
    std = float(std) if std and std > 1 else 1
    offset = get_offset(offset)

    # Calculate Result
    std *= stdev(close=close, length=length, **kwargs)
    mean = sma(close=close, length=length, **kwargs)
    zscore = (close - mean) / std

    # Offset
    zscore = zscore.shift(offset)

    # Name & Category
    zscore.name = f"Z_{length}"
    zscore.category = 'statistics'

    return zscore