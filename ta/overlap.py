# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from math import sqrt
from .utils import get_drift, get_offset, verify_series
# from .volatility import 


def hl2(high, low, offset=None, **kwargs):
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


def hlc3(high, low, close, offset=None, **kwargs):
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


def ohlc4(open_, high, low, close, offset=None, **kwargs):
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


def midpoint(close, length=None, offset=None, **kwargs):
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


def midprice(high, low, length=None, offset=None, **kwargs):
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


def dema(close, length=None, offset=None, **kwargs):
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


def ema(close, length=None, offset=None, **kwargs):
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


def hma(close, length=None, offset=None, **kwargs):
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


def rma(close, length=None, offset=None, **kwargs):
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


def sma(close, length=None, offset=None, **kwargs):
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


def t3(close, length=None, a=None, offset=None, **kwargs):
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


def tema(close, length=None, offset=None, **kwargs):
    """Indicator: Triple Exponential Moving Average (TEMA)"""
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


def trima(close, length=None, offset=None, **kwargs):
    """Indicator: Triangular Moving Average (TRIMA)  *requires scipy"""
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


def vwap(high, low, close, volume, offset=None, **kwargs):
    """Indicator: Volume Weighted Average Price (VWAP)"""
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


def vwma(close, volume, length=None, offset=None, **kwargs):
    """Indicator: Volume Weighted Moving Average (VWMA)"""
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


def wma(close, length=None, asc=None, offset=None, **kwargs):
    """Indicator: Weighted Moving Average (WMA)"""
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



# Overlap Documentation
hl2_docs = \
"""Average of High-Low (HL2)

Equally weighted Average of two series', namely High and Low.

Sources:
    https://www.tradingview.com/study-script-reference/#var_hl2

Calculation:
    HL2 = 0.5 * (high + low)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


hlc3_docs = \
"""Average of High-Low-Close (HLC3)

Equally weighted Average of three series', namely High, Low, Close.

Sources:
    https://www.tradingview.com/study-script-reference/#var_hlc3

Calculation:
    HLC3 = (high + low + close) / 3.0

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


ohlc4_docs = \
"""Average of Open-High-Low-Close (OHLC4)

Equally weighted Average of four series', namely Open, High, Low, Close.

Sources:
    https://www.tradingview.com/study-script-reference/#var_ohlc4

Calculation:
    OHLC4 = 0.25 * (open + high + low + close)

Args:
    open (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


midpoint_docs = \
"""Midpoint (MIDPOINT)

The Midpoint is the average of the highest and lowest closes over a period.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/midpoint-midpnt/

Calculation:
    Default Inputs:
        length=1
    lowest_close  = close.rolling(length).min()
    highest_close = close.rolling(length).max()

    MIDPOINT = 0.5 * (highest_close + lowest_close)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 14
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


midprice_docs = \
"""Midprice (MIDPRICE)

William's Percent R is a momentum oscillator similar to the RSI that
attempts to identify overbought and oversold conditions.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/midprice-midpri/

Calculation:
    Default Inputs:
        length=1
    lowest_low   = low.rolling(length).min()
    highest_high = high.rolling(length).max()

    MIDPRICE = 0.5 * (highest_high + lowest_low)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    length (int): It's period.  Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


dema_docs = \
"""Double Exponential Moving Average (DEMA)

The Double Exponential Moving Average attempts to a smoother average with less
lag than the normal Exponential Moving Average (EMA).

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/double-exponential-moving-average-dema/

Calculation:
    Default Inputs:
        length=10
    EMA = Exponential Moving Average
    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)

    DEMA = 2 * ema1 - ema2

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


ema_docs = \
"""Exponential Moving Average (DEMA)

The Exponential Moving Average is more responsive moving average compared to the
Simple Moving Average (SMA).  The weights are determined by alpha which is
proportional to it's length.  There are several different methods of calculating
EMA.  One method uses just the standard definition of EMA and another uses the
SMA to generate the initial value for the rest of the calculation.

Sources:
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:moving_averages
    https://www.investopedia.com/ask/answers/122314/what-exponential-moving-average-ema-formula-and-how-ema-calculated.asp

Calculation:
    Default Inputs:
        length=10
    SMA = Simple Moving Average
    If 'presma':
        initial = SMA(close, length)
        rest = close[length:]
        close = initial + rest

    EMA = close.ewm(span=length, adjust=adjust).mean()

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    adjust (bool): Default: True
    presma (bool, optional): If True, uses SMA for initial value.
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


hma_docs = \
"""Hull Moving Average (HMA)

The Hull Exponential Moving Average attempts to reduce or remove lag in moving
averages.

Sources:
    https://alanhull.com/hull-moving-average

Calculation:
    Default Inputs:
        length=10
    WMA = Weighted Moving Average
    half_length = int(0.5 * length)
    sqrt_length = int(math.sqrt(length))

    wmaf = WMA(close, half_length)
    wmas = WMA(close, length)
    HMA = WMA(2 * wmaf - wmas, sqrt_length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


rma_docs = \
"""wildeR's Moving Average (RMA)

The WildeR's Moving Average is simply an Exponential Moving Average (EMA)
with a modified alpha = 1 / length.

Sources:
    https://alanhull.com/hull-moving-average

Calculation:
    Default Inputs:
        length=10
    EMA = Exponential Moving Average
    alpha = 1 / length
    RMA = EMA(close, alpha=alpha)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


sma_docs = \
"""Simple Moving Average (SMA)

The Simple Moving Average is the classic moving average that is the equally
weighted average over n periods.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/simple-moving-average-sma/

Calculation:
    Default Inputs:
        length=10
    SMA = SUM(close, length) / length

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    adjust (bool): Default: True
    presma (bool, optional): If True, uses SMA for initial value.
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


t3_docs = \
"""Tim Tillson's T3 Moving Average (T3)

Tim Tillson's T3 Moving Average is considered a smoother and more responsive
moving average relative to other moving averages.

Sources:
    http://www.binarytribune.com/forex-trading-indicators/t3-moving-average-indicator/

Calculation:
    Default Inputs:
        length=10, a=0.7
    c1 = -a^3
    c2 = 3a^2 + 3a^3 = 3a^2 * (1 + a)
    c3 = -6a^2 - 3a - 3a^3
    c4 = a^3 + 3a^2 + 3a + 1

    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)
    ema3 = EMA(ema2, length)
    ema4 = EMA(ema3, length)
    ema5 = EMA(ema4, length)
    ema6 = EMA(ema5, length)
    T3 = c1 * ema6 + c2 * ema5 + c3 * ema4 + c4 * ema3

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    a (float): 0 < a < 1.  Default: 0.7
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    adjust (bool): Default: True
    presma (bool, optional): If True, uses SMA for initial value.
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


tema_docs = \
"""Triple Exponential Moving Average (TEMA)

A less laggy Exponential Moving Average.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/triple-exponential-moving-average-tema/

Calculation:
    Default Inputs:
        length=10
    EMA = Exponential Moving Average
    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)
    ema3 = EMA(ema2, length)
    TEMA = 3 * (ema1 - ema2) + ema3

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    adjust (bool): Default: True
    presma (bool, optional): If True, uses SMA for initial value.
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


trima_docs = \
"""Triangular Moving Average (TRIMA)
REQUIRES: scipy

A weighted moving average where the shape of the weights are triangular and the
greatest weight is in the middle of the period.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/triangular-moving-average-trima/

Calculation:
    Default Inputs:
        length=10
    SMA = Simple Moving Average
    if scipy:
        TRIMA = close.rolling(length, win_type='triang').mean()
    else:
        half_length = math.round(0.5 * (length + 1))
        SMA1 = SUM(close, half_length) / half_length
        TRIMA = SMA(SMA1, half_length) / half_length

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    adjust (bool): Default: True
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


vwap_docs = \
"""Volume Weighted Average Price (VWAP)

The Volume Weighted Average Price that measures the average typical price
by volume.  It is typically used with intraday charts to identify general
direction.

Sources:
    https://www.tradingview.com/wiki/Volume_Weighted_Average_Price_(VWAP)
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/volume-weighted-average-price-vwap/

Calculation:
    tp = typical_price = hlc3(high, low, close)
    tpv = tp * volume
    VWAP = tpv.cumsum() / volume.cumsum()

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


vwma_docs = \
"""Volume Weighted Moving Average (VWMA)

Volume Weighted Moving Average.

Sources:
    https://www.motivewave.com/studies/volume_weighted_moving_average.htm

Calculation:
    Default Inputs:
        length=10
    SMA = Simple Moving Average
    pv = close * volume
    VWMA = SMA(pv, length) / SMA(volume, length)

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): It's period.  Default: 10
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


wma_docs = \
"""Weighted Moving Average (WMA)

The Weighted Moving Average where the weights are linearly increasing and
the most recent data has the heaviest weight.

Sources:
    https://en.wikipedia.org/wiki/Moving_average#Weighted_moving_average

Calculation:
    Default Inputs:
        length=10, asc=True
    total_weight = 0.5 * length * (length + 1)
    weights_ = [1, 2, ..., length + 1]  # Ascending
    weights = weights if asc else weights[::-1]

    def linear_weights(w):
        def _compute(x):
            return (w * x).sum() / total_weight
        return _compute

    WMA = close.rolling(length)_.apply(linear_weights(weights), raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 10
    asc (bool): Recent values weigh more.  Default: True
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""



# Overlap Documentation
hl2.__doc__ = hl2_docs
hlc3.__doc__ = hlc3_docs
ohlc4.__doc__ = ohlc4_docs
midpoint.__doc__ = midpoint_docs
midprice.__doc__ = midprice_docs
dema.__doc__ = dema_docs
ema.__doc__ = ema_docs
hma.__doc__ = hma_docs
rma.__doc__ = rma_docs
sma.__doc__ = sma_docs
t3.__doc__ = t3_docs
tema.__doc__ = tema_docs
trima.__doc__ = trima_docs
vwap.__doc__ = vwap_docs
vwma.__doc__ = vwma_docs
wma.__doc__ = wma_docs