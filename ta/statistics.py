# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .overlap import vwma
from .utils import get_offset, verify_series


def kurtosis(close:pd.Series, length=None, offset=None, **kwargs):
    """Kurtosis over periods of a Pandas Series
    
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
    """Moving Covariance over periods of a Pandas Series
    
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
    # mean = sma(vwap, length=length)
    std = stdev(close=vwap, length=length)
    mcv = 100 * std / mean

    # Offset
    mcv = mcv.shift(offset)

    # Name & Category
    mcv.name = f"MCV_{length}"
    mcv.category = 'statistics'

    return mcv


def quantile(close:pd.Series, length=None, q=None, offset=None, **kwargs):
    """Quantile over periods of a Pandas Series
    
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
    """Skew over periods of a Pandas Series
    
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
    """Standard Deviation over periods of a Pandas Series
    
    Use help(df.ta.stdev) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 30
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    stdev = variance(close, length=length).apply(np.sqrt)

    # Offset
    stdev = stdev.shift(offset)

    # Name & Category
    stdev.name = f"STDEV_{length}"
    stdev.category = 'statistics'

    return stdev


def variance(close:pd.Series, length=None, offset=None, **kwargs):
    """Variance over periods of a Pandas Series
    
    Use help(df.ta.stdev) for specific documentation where 'df' represents
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

