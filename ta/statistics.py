# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .overlap import ema, sma, vwma
from .utils import get_offset, verify_series


def kurtosis(close, length=None, offset=None, **kwargs):
    """Indicator: Kurtosis"""
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


def median(close, length=None, offset=None, **kwargs):
    """Indicator: Median"""
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 30
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    median = close.rolling(length, min_periods=min_periods).median()

    # Offset
    median = median.shift(offset)

    # Name & Category
    median.name = f"MEDIAN_{length}"
    median.category = 'statistics'

    return median


def quantile(close, length=None, q=None, offset=None, **kwargs):
    """Indicator: Quantile"""
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


def skew(close, length=None, offset=None, **kwargs):
    """Indicator: Skew"""
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


def stdev(close, length=None, offset=None, **kwargs):
    """Indicator: Standard Deviation"""
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


def variance(close, length=None, offset=None, **kwargs):
    """Indicator: Variance"""
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


def zscore(close, length=None, std=None, offset=None, **kwargs):
    """Indicator: Z Score"""
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



# Statistics Documentation
kurtosis_docs = \
"""Kurtosis

Rolling Kurtosis

Sources:

Calculation:
    Default Inputs:
        length=30
    KURTOSIS = close.rolling(length).kurt()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 30
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


median_docs = \
"""Median

Rolling Median of over 'n' periods.  Sibling of a Simple Moving Average.

Sources:
    https://www.incrediblecharts.com/indicators/median_price.php

Calculation:
    Default Inputs:
        length=1
    MEDIAN = close.rolling(length).median()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 30
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


quantile_docs = \
"""Quantile

Rolling Quantile

Sources:

Calculation:
    Default Inputs:
        length=30, q=0.5
    QUANTILE = close.rolling(length).quantile(q)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 30
    q (float): It's period.  Default: 0.5
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


skew_docs = \
"""Skew

Rolling Skew

Sources:

Calculation:
    Default Inputs:
        length=30
    SKEW = close.rolling(length).skew()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 30
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


stdev_docs = \
"""Standard Deviation

Rolling Standard Deviation

Sources:

Calculation:
    Default Inputs:
        length=30
    VAR = Variance
    STDEV = variance(close, length).apply(np.sqrt)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 30
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


variance_docs = \
"""Variance

Rolling Variance

Sources:

Calculation:
    Default Inputs:
        length=30
    VARIANCE = close.rolling(length).var()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 30
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


zscore_docs = \
"""ZScore

Rolling Z Score

Sources:

Calculation:
    Default Inputs:
        length=30, std=1
    SMA = Simple Moving Average
    STDEV = Standard Deviation
    std = std * STDEV(close, length)
    mean = SMA(close, length)
    ZSCORE = (close - mean) / std

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 30
    std (float): It's period.  Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


# Statistics Documentation
kurtosis.__doc__ = kurtosis_docs
median.__doc__ = median_docs
quantile.__doc__ = quantile_docs
skew.__doc__ = skew_docs
stdev.__doc__ = stdev_docs
variance.__doc__ = variance_docs
zscore.__doc__ = zscore_docs