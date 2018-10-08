# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from math import sqrt
from .utils import get_drift, get_offset, verify_series
# from .volatility import 


def hl2(high:pd.Series, low:pd.Series, offset=None, **kwargs):
    """Indicator: HL2 """
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
    """Indicator: HLC3"""
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

    return hlc3


def ohlc4(open_:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series, offset=None, **kwargs):
    """Indicator: OHLC4"""
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

    return ohlc4


def midpoint(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Midpoint"""
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

    return midpoint


def midprice(high:pd.Series, low:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Midprice"""
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

    return midprice


def dema(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Double Exponential Moving Average (DEMA)"""
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 10
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    ema1 = ema(close=close, length=length, min_periods=min_periods)
    ema2 = ema(close=ema1, length=length, min_periods=min_periods)
    dema = 2 * ema1 - ema2

    # Offset
    dema = dema.shift(offset)

    # Name & Category
    dema.name = f"DEMA_{length}"
    dema.category = 'overlap'

    return dema


def ema(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Exponential Moving Average (EMA)"""
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 10
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    adjust = bool(kwargs['adjust']) if 'adjust' in kwargs and kwargs['adjust'] is not None else True
    offset = get_offset(offset)

    # Calculate Result
    if 'presma' in kwargs and kwargs['presma']:
        initial_sma = sma(close=close, length=length)[:length]
        rest = close[length:]
        close = pd.concat([initial_sma, rest])

    ema = close.ewm(span=length, min_periods=min_periods, adjust=adjust).mean()

    # Offset
    ema = ema.shift(offset)

    # Name & Category
    ema.name = f"EMA_{length}"
    ema.category = 'overlap'

    return ema


def hma(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Hull Moving Average (HMA)
    
    Use help(df.ta.hma) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 10
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    half_length = int(length / 2)
    sqrt_length = int(sqrt(length))

    wmaf = wma(close=close, length=half_length)
    wmas = wma(close=close, length=length)
    hma = wma(close=2 * wmaf - wmas, length=sqrt_length)

    # Offset
    hma = hma.shift(offset)

    # Name & Category
    hma.name = f"HMA_{length}"
    hma.category = 'overlap'

    return hma


def rma(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: wildeR's Moving Average (RMA)"""
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 10
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)
    alpha = (1.0 / length) if length > 0 else 1

    # Calculate Result
    rma = close.ewm(alpha=alpha, min_periods=min_periods).mean()

    # Offset
    rma = rma.shift(offset)

    # Name & Category
    rma.name = f"RMA_{length}"
    rma.category = 'overlap'

    return rma


def sma(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Simple Moving Average (SMA)"""
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 10
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    sma = close.rolling(length, min_periods=min_periods).mean()

    # Offset
    sma = sma.shift(offset)

    # Name & Category
    sma.name = f"SMA_{length}"
    sma.category = 'overlap'

    return sma


def t3(close:pd.Series, length=None, a=None, offset=None, **kwargs):
    """Indicator: T3"""
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 10
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    a = float(a) if a and a > 0 and a < 1 else 0.7
    offset = get_offset(offset)

    # Calculate Result
    c1 = -a * a ** 2
    c2 = 3 * a ** 2 + 3 * a ** 3
    c3 = -6 * a ** 2 - 3 * a - 3 * a ** 3
    c4 = a ** 3 + 3 * a ** 2 + 3 * a + 1

    e1 = ema(close=close, length=length, min_periods=min_periods, **kwargs)
    e2 = ema(close=e1, length=length, min_periods=min_periods, **kwargs)
    e3 = ema(close=e2, length=length, min_periods=min_periods, **kwargs)
    e4 = ema(close=e3, length=length, min_periods=min_periods, **kwargs)
    e5 = ema(close=e4, length=length, min_periods=min_periods, **kwargs)
    e6 = ema(close=e5, length=length, min_periods=min_periods, **kwargs)
    t3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3

    # Offset
    t3 = t3.shift(offset)

    # Name & Category
    t3.name = f"T3_{length}_{a}"
    t3.category = 'overlap'

    return t3


def tema(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Triple Exponential Moving Average (TEMA)
    
    Use help(df.ta.tema) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 10
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    ema1 = ema(close=close, length=length, min_periods=min_periods)
    ema2 = ema(close=ema1, length=length, min_periods=min_periods)
    ema3 = ema(close=ema2, length=length, min_periods=min_periods)
    tema = 3 * (ema1 - ema2) + ema3

    # Offset
    tema = tema.shift(offset)

    # Name & Category
    tema.name = f"TEMA_{length}"
    tema.category = 'overlap'

    return tema


def trima(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Triangular Moving Average (TRIMA), requires scipy
    
    Use help(df.ta.trima) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 10
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    trima = close.rolling(length, min_periods=min_periods, win_type='triang').mean()

    # Offset
    trima = trima.shift(offset)

    # Name & Category
    trima.name = f"TRIMA_{length}"
    trima.category = 'overlap'

    return trima


def vwap(high:pd.Series, low:pd.Series, close:pd.Series, volume:pd.Series, offset=None, **kwargs):
    """Indicator: Volume Weighted Average Price (VWAP)
    
    Use help(df.ta.vwap) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    volume = verify_series(volume)
    offset = get_offset(offset)

    # Calculate Result
    tp = hlc3(high=high, low=low, close=close)
    tpv = tp * volume
    vwap = tpv.cumsum() / volume.cumsum()

    # Offset
    vwap = vwap.shift(offset)

    # Name & Category
    vwap.name = "VWAP"
    vwap.category = 'overlap'

    return vwap


def vwma(close:pd.Series, volume:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Volume Weighted Moving Average (VWMA)
    
    Use help(df.ta.vwma) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    volume = verify_series(volume)
    length = int(length) if length and length > 0 else 10
    offset = get_offset(offset)

    # Calculate Result
    pv = close * volume
    vwma = sma(close=pv, length=length) / sma(close=volume, length=length)

    # Offset
    vwma = vwma.shift(offset)

    # Name & Category
    vwma.name = f"VWMA_{length}"
    vwma.category = 'overlap'

    return vwma


def wma(close:pd.Series, length=None, asc=None, offset=None, **kwargs):
    """Indicator: Weighted Moving Average (WMA)
    
    Use help(df.ta.wma) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 10
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    asc = asc if asc else True
    offset = get_offset(offset)

    # Calculate Result
    total_weight = 0.5 * length * (length + 1)
    weights_ = pd.Series(np.arange(1, length + 1))
    weights = weights_ if asc else weights_[::-1]

    def linear_weights(w):
        def _compute(x):
            return (w * x).sum() / total_weight
        return _compute

    close_ = close.rolling(length, min_periods=length)
    wma =close_.apply(linear_weights(weights), raw=True)

    # Offset
    wma = wma.shift(offset)

    # Name & Category
    wma.name = f"WMA_{length}"
    wma.category = 'overlap'

    return wma