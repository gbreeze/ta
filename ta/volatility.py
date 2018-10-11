# -*- coding: utf-8 -*-
"""
.. module:: volatility
   :synopsis: Volatility Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd

from .utils import *
from .overlap import ema, hlc3
from .statistics import variance, stdev


def atr(high:pd.Series, low:pd.Series, close:pd.Series, length=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: Average True Range (ATR)"""
    # Validate arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    length = int(length) if length and length > 0 else 14
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    mamode = mamode.lower() if mamode else 'ema'
    drift = get_drift(drift)
    offset = get_offset(offset)

    # Calculate Result
    tr = true_range(high=high, low=low, close=close, drift=drift)
    if mamode == 'ema':
        atr = tr.ewm(span=length, min_periods=min_periods).mean()
    else:
        atr = tr.rolling(length, min_periods=min_periods).mean()

    # Offset
    atr = atr.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        atr.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        atr.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    atr.name = f"ATR_{length}"
    atr.category = 'volatility'

    return atr


def bbands(close:pd.Series, length=None, std=None, mamode=None, offset=None, **kwargs):
    """Indicator: Bollinger Bands (BBANDS)"""
    # Validate arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 20
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    std = float(std) if std and std > 0 else 2
    mamode = mamode.lower() if mamode else 'ema'
    offset = get_offset(offset)

    # Calculate Result
    standard_deviation = stdev(close=close, length=length)
    # std = variance(close=close, length=length).apply(np.sqrt)

    if mamode is None or mamode == 'sma':
        mid = close.rolling(length, min_periods=min_periods).mean()
    elif mamode == 'ema':
        mid = close.ewm(span=length, min_periods=min_periods).mean()

    lower = mid - std * standard_deviation
    upper = mid + std * standard_deviation

    # Offset
    lower = lower.shift(offset)
    mid = mid.shift(offset)
    upper = upper.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        lower.fillna(kwargs['fillna'], inplace=True)
        mid.fillna(kwargs['fillna'], inplace=True)
        upper.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        lower.fillna(method=kwargs['fill_method'], inplace=True)
        mid.fillna(method=kwargs['fill_method'], inplace=True)
        upper.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    lower.name = f"BBL_{length}"
    mid.name = f"BBM_{length}"
    upper.name = f"BBU_{length}"
    mid.category = upper.category = lower.category = 'volatility'

    # Prepare DataFrame to return
    data = {lower.name: lower, mid.name: mid, upper.name: upper}
    bbandsdf = pd.DataFrame(data)
    bbandsdf.name = f"BBANDS_{length}"
    bbandsdf.category = 'volatility'

    return bbandsdf


def donchian(close:pd.Series, length=None, offset=None, **kwargs):
    """Indicator: Donchian Channels (DC)"""
    # Validate arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 20
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    lower = close.rolling(length, min_periods=min_periods).min()
    upper = close.rolling(length, min_periods=min_periods).max()
    mid = 0.5 * (lower + upper)

    # Handle fills
    if 'fillna' in kwargs:
        lower.fillna(kwargs['fillna'], inplace=True)
        mid.fillna(kwargs['fillna'], inplace=True)
        upper.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        lower.fillna(method=kwargs['fill_method'], inplace=True)
        mid.fillna(method=kwargs['fill_method'], inplace=True)
        upper.fillna(method=kwargs['fill_method'], inplace=True)

    # Offset
    lower = lower.shift(offset)
    mid = mid.shift(offset)
    upper = upper.shift(offset)

    # Name and Categorize it
    lower.name = f"DCL_{length}"
    mid.name = f"DCM_{length}"
    upper.name = f"DCU_{length}"
    mid.category = upper.category = lower.category = 'volatility'

    # Prepare DataFrame to return
    data = {lower.name: lower, mid.name: mid, upper.name: upper}
    dcdf = pd.DataFrame(data)
    dcdf.name = f"DC_{length}"
    dcdf.category = 'volatility'

    return dcdf


def kc(high:pd.Series, low:pd.Series, close:pd.Series, length=None, scalar=None, mamode=None, offset=None, **kwargs):
    """Indicator: Keltner Channels (KC)"""
    # Validate arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    length = int(length) if length and length > 0 else 20
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    scalar = float(scalar) if scalar and scalar > 0 else 2
    mamode = mamode.lower() if mamode else None
    offset = get_offset(offset)

    # Calculate Result
    std = variance(close=close, length=length).apply(np.sqrt)

    if mamode == 'ema':
        basis = close.ewm(span=length, min_periods=min_periods).mean()
        band = atr(high=high, low=low, close=close)
    else:
        hl_range = high - low
        typical_price = hlc3(high=high, low=low, close=close)
        basis = typical_price.rolling(length, min_periods=min_periods).mean()
        band = hl_range.rolling(length, min_periods=min_periods).mean()

    lower = basis - scalar * band
    upper = basis + scalar * band

    # Offset
    lower = lower.shift(offset)
    basis = basis.shift(offset)
    upper = upper.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        lower.fillna(kwargs['fillna'], inplace=True)
        basis.fillna(kwargs['fillna'], inplace=True)
        upper.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        lower.fillna(method=kwargs['fill_method'], inplace=True)
        basis.fillna(method=kwargs['fill_method'], inplace=True)
        upper.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    lower.name = f"KCL_{length}"
    basis.name = f"KCB_{length}"
    upper.name = f"KCU_{length}"
    basis.category = upper.category = lower.category = 'volatility'

    # Prepare DataFrame to return
    data = {lower.name: lower, basis.name: basis, upper.name: upper}
    kcdf = pd.DataFrame(data)
    kcdf.name = f"KC_{length}"
    kcdf.category = 'volatility'

    return kcdf


def massi(high:pd.Series, low:pd.Series, fast=None, slow=None, offset=None, **kwargs):
    """Indicator: Mass Index (MASSI)"""
    # Validate arguments
    high = verify_series(high)
    low = verify_series(low)
    fast = int(fast) if fast and fast > 0 else 9
    slow = int(slow) if slow and slow > 0 else 25
    if slow < fast:
        fast, slow = slow, fast
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else fast
    offset = get_offset(offset)

    # Calculate Result
    hl_range = high - low
    hl_ema1 = ema(close=hl_range, length=fast)
    hl_ema2 = ema(close=hl_ema1, length=fast)

    hl_ratio = hl_ema1 / hl_ema2
    massi = hl_ratio.rolling(slow, min_periods=slow).sum()

    # Offset
    massi = massi.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        massi.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        massi.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    massi.name = f"MASSI_{fast}_{slow}"
    massi.category = 'volatility'

    return massi


def natr(high:pd.Series, low:pd.Series, close:pd.Series, length=None, mamode=None, drift=None, offset=None, **kwargs):
    """Indicator: Normalized Average True Range (NATR)"""
    # Validate arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    length = int(length) if length and length > 0 else 14
    mamode = mamode.lower() if mamode else 'ema'
    drift = get_drift(drift)
    offset = get_offset(offset)

    # Calculate Result
    natr = (100 / close) * atr(high=high, low=low, close=close, length=length, mamode=mamode, drift=drift, offset=offset, **kwargs)

    # Offset
    natr = natr.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        natr.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        natr.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    natr.name = f"NATR_{length}"
    natr.category = 'volatility'

    return natr


def true_range(high:pd.Series, low:pd.Series, close:pd.Series, drift=None, offset=None, **kwargs):
    """Indicator: True Range"""
    # Validate arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    drift = get_drift(drift)
    offset = get_offset(offset)

    # Calculate Result
    prev_close = close.shift(drift)
    ranges = [high - low, high - prev_close, low - prev_close]
    true_range = pd.DataFrame(ranges).T
    true_range = true_range.abs().max(axis=1)

    # Offset
    true_range = true_range.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        true_range.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        true_range.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    true_range.name = f"TRUERANGE_{drift}"
    true_range.category = 'volatility'

    return true_range



# Legacy Code
def average_true_range(high, low, close, n=14, fillna=False):
    """Average True Range (ATR)

    The indicator provide an indication of the degree of price volatility.
    Strong moves, in either direction, are often accompanied by large ranges,
    or large True Ranges.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:average_true_range_atr

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    cs = close.shift(1)
    tr = high.combine(cs, max) - low.combine(cs, min)
    tr = ema(tr, n)
    if fillna:
        tr = tr.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(tr, name='atr')


def bollinger_mavg(close, n=20, fillna=False):
    """Bollinger Bands (BB)

    N-period simple moving average (MA).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n).mean()
    if fillna:
        mavg = mavg.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(mavg, name='mavg')


def bollinger_hband(close, n=20, ndev=2, fillna=False):
    """Bollinger Bands (BB)

    Upper band at K times an N-period standard deviation above the moving
    average (MA + Kdeviation).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation

    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev*mstd
    if fillna:
        hband = hband.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(hband, name='hband')


def bollinger_lband(close, n=20, ndev=2, fillna=False):
    """Bollinger Bands (BB)

    Lower band at K times an N-period standard deviation below the moving
    average (MA − Kdeviation).

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation

    Returns:
        pandas.Series: New feature generated.
    """
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev * mstd
    if fillna:
        lband = lband.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(lband, name='lband')


def bollinger_hband_indicator(close, n=20, ndev=2, fillna=False):
    """Bollinger High Band Indicator

    Returns 1, if close is higher than bollinger high band. Else, return 0.

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    hband = mavg + ndev * mstd
    df['hband'] = 0.0
    df.loc[close > hband, 'hband'] = 1.0
    hband = df['hband']
    if fillna:
        hband = hband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(hband, name='bbihband')


def bollinger_lband_indicator(close, n=20, ndev=2, fillna=False):
    """Bollinger Low Band Indicator

    Returns 1, if close is lower than bollinger low band. Else, return 0.

    https://en.wikipedia.org/wiki/Bollinger_Bands

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        ndev(int): n factor standard deviation

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    mavg = close.rolling(n).mean()
    mstd = close.rolling(n).std()
    lband = mavg - ndev * mstd
    df['lband'] = 0.0
    df.loc[close < lband, 'lband'] = 1.0
    lband = df['lband']
    if fillna:
        lband = lband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(lband, name='bbilband')


def keltner_channel_central(high, low, close, n=10, fillna=False):
    """Keltner channel (KC)

    Showing a simple moving average line (central) of typical price.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    tp = (high + low + close) / 3.0
    tp = tp.rolling(n).mean()
    if fillna:
        tp = tp.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(tp, name='kc_central')


def keltner_channel_hband(high, low, close, n=10, fillna=False):
    """Keltner channel (KC)

    Showing a simple moving average line (high) of typical price.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    tp = ((4 * high) - (2 * low) + close) / 3.0
    tp = tp.rolling(n).mean()
    if fillna:
        tp = tp.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(tp, name='kc_hband')


def keltner_channel_lband(high, low, close, n=10, fillna=False):
    """Keltner channel (KC)

    Showing a simple moving average line (low) of typical price.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    tp = ((-2 * high) + (4 * low) + close) / 3.0
    tp = tp.rolling(n).mean()
    if fillna:
        tp = tp.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(tp, name='kc_lband')


def keltner_channel_hband_indicator(high, low, close, n=10, fillna=False):
    """Keltner Channel High Band Indicator (KC)

    Returns 1, if close is higher than keltner high band channel. Else,
    return 0.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = ((4 * high) - (2 * low) + close) / 3.0
    df.loc[close > hband, 'hband'] = 1.0
    hband = df['hband']
    if fillna:
        hband = hband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(hband, name='kci_hband')


def keltner_channel_lband_indicator(high, low, close, n=10, fillna=False):
    """Keltner Channel Low Band Indicator (KC)

    Returns 1, if close is lower than keltner low band channel. Else, return 0.

    https://en.wikipedia.org/wiki/Keltner_channel

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = ((-2 * high) + (4 * low) + close) / 3.0
    df.loc[close < lband, 'lband'] = 1.0
    lband = df['lband']
    if fillna:
        lband = lband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(lband, name='kci_lband')


def donchian_channel_hband(close, n=20, fillna=False):
    """Donchian channel (DC)

    The upper band marks the highest price of an issue for n periods.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    hband = close.rolling(n).max()
    if fillna:
        hband = hband.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(hband, name='dchband')


def donchian_channel_lband(close, n=20, fillna=False):
    """Donchian channel (DC)

    The lower band marks the lowest price for n periods.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    lband = close.rolling(n).min()
    if fillna:
        lband = lband.replace([np.inf, -np.inf], np.nan).fillna(method='backfill')
    return pd.Series(lband, name='dclband')


def donchian_channel_hband_indicator(close, n=20, fillna=False):
    """Donchian High Band Indicator

    Returns 1, if close is higher than donchian high band channel. Else,
    return 0.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['hband'] = 0.0
    hband = close.rolling(n).max()
    df.loc[close >= hband, 'hband'] = 1.0
    hband = df['hband']
    if fillna:
        hband = hband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(hband, name='dcihband')


def donchian_channel_lband_indicator(close, n=20, fillna=False):
    """Donchian Low Band Indicator

    Returns 1, if close is lower than donchian low band channel. Else, return 0.

    https://www.investopedia.com/terms/d/donchianchannels.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close]).transpose()
    df['lband'] = 0.0
    lband = close.rolling(n).min()
    df.loc[close <= lband, 'lband'] = 1.0
    lband = df['lband']
    if fillna:
        lband = lband.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(lband, name='dcilband')
