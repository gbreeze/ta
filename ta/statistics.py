# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

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

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[kurtosis.name] = kurtosis

    return kurtosis


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

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[quantile.name] = quantile

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

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[skew.name] = skew

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

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[stdev.name] = stdev

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

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[variance.name] = variance

    return variance

