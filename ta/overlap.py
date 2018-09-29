# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .utils import get_offset, verify_series


def hl2(high:pd.Series, low:pd.Series, offset=None, **kwargs):
    """HL2 of a Pandas Series
    
    Use help(df.ta.hl2) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    offset = get_offset(offset)

    # Calculate Result
    hl2 = 0.5 * (high + low)

    # Offset
    hl2 = hl2.shift(offset)

    # Name & Category
    hl2.name = "HL2"
    hl2.category = 'overlap'

    return hl2


def hlc3(high:pd.Series, low:pd.Series, close:pd.Series, offset=None, **kwargs):
    """HLC3 of a Pandas Series
    
    Use help(df.ta.hlc3) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)

    # Calculate Result
    hlc3 = (high + low + close) / 3

    # Offset
    hlc3 = hlc3.shift(offset)

    # Name & Category
    hlc3.name = "HLC3"
    hlc3.category = 'overlap'

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[hlc3.name] = hlc3

    return hlc3


def ohlc4(open_:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series, offset=None, **kwargs):
    """OHLC4 of a Pandas Series
    
    Use help(df.ta.ohlc4) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)

    # Calculate Result
    ohlc4 = 0.25 * (open_ + high + low + close)

    # Offset
    ohlc4 = ohlc4.shift(offset)

    # Name & Category
    ohlc4.name = "OHLC4"
    ohlc4.category = 'overlap'

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[ohlc4.name] = ohlc4

    return ohlc4


def median(close:pd.Series, length=None, offset=None, **kwargs):
    """Median of a Pandas Series
    
    Use help(df.ta.median) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 5
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    median = close.rolling(length, min_periods=min_periods).median()

    # Offset
    median = median.shift(offset)

    # Name & Category
    median.name = f"MEDIAN_{length}"
    median.category = 'overlap'

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[median.name] = median

    return median


def midpoint(close:pd.Series, length=None, offset=None, **kwargs):
    """Midpoint of a Pandas Series
    
    Use help(df.ta.midpoint) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 1
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    lowest = close.rolling(length, min_periods=min_periods).min()
    highest = close.rolling(length, min_periods=min_periods).max()
    midpoint = 0.5 * (lowest + highest)

    # Offset
    midpoint = midpoint.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        midpoint.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        midpoint.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    midpoint.name = f"MIDPOINT_{length}"
    midpoint.category = 'overlap'

    # If append, then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[midpoint.name] = midpoint

    return midpoint


def midprice(high:pd.Series, low:pd.Series, length=None, offset=None, **kwargs):
    """Midprice of a Pandas Series
    
    Use help(df.ta.midprice) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate arguments
    high = verify_series(high)
    low = verify_series(low)
    length = int(length) if length and length > 0 else 1
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    lowest_low = low.rolling(length, min_periods=min_periods).min()
    highest_high = high.rolling(length, min_periods=min_periods).max()
    midprice = 0.5 * (lowest_low + highest_high)

    # Offset
    midprice = midprice.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        midprice.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        midprice.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    midprice.name = f"MIDPRICE_{length}"
    midprice.category = 'overlap'

    # If append, then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[midprice.name] = midprice

    return midprice


def rpn(high:pd.Series, low:pd.Series, length=None, offset=None, percentage=None, **kwargs):
    """Percent of Range of a Pandas Series
    
    Use help(df.ta.rpn) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate arguments
    high = verify_series(high)
    low = verify_series(low)
    length = int(length) if length and length > 0 else 1
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    percentage = float(percentage) if percentage and percentage > 0 and percentage < 100 else 0.1

    # Calculate Result
    highest_high = high.rolling(length, min_periods=min_periods).max()
    lowest_low = low.rolling(length, min_periods=min_periods).min()
    abs_range = (highest_high - lowest_low).abs()
    rp = percentage * abs_range

    if 'withLow' in kwargs and kwargs['withLow']:
        rp += low

    # Name & Category
    rp.name = f"RP_{length}_{percentage}"
    rp.category = 'overlap'

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[rp.name] = rp

    return rp