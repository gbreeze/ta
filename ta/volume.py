# -*- coding: utf-8 -*-
"""
.. module:: volume
   :synopsis: Volume Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd

from .utils import get_drift, get_offset, signed_series, verify_series
from .momentum import roc
from .overlap import hl2


def ad(high:pd.Series, low:pd.Series, close:pd.Series, volume:pd.Series, open_:pd.Series, signed=True, offset=None, **kwargs):
    """Accumulation/Distribution of a Pandas Series
    
    Use help(df.ta.ad) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    volume = verify_series(volume)
    offset = get_offset(offset)

    # Calculate Result
    if open_ is not None:
        open_ = verify_series(open_)
        ad = close - open_  # AD with Open
    else:                
        ad = 2 * close - high - low  # AD with High, Low, Close

    hl_range = high - low
    ad *= volume / hl_range
    ad = ad.cumsum()

    # Offset
    ad = ad.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        ad.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        ad.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    ad.name = f"AD"
    ad.category = 'volume'

    return ad


def cmf(high:pd.Series, low:pd.Series, close:pd.Series, volume:pd.Series, open_:pd.Series, length=None, offset=None, **kwargs):
    """Chaikin Money Flow (CMF)
    
    Use help(df.ta.cmf) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    volume = verify_series(volume)
    length = int(length) if length and length > 0 else 1
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    if open_ is not None:
        open_ = verify_series(open_)
        ad = close - open_  # AD with Open
    else:                
        ad = 2 * close - high - low  # AD with High, Low, Close

    hl_range = high - low
    ad *= volume / hl_range
    cmf = ad.rolling(length).sum() / volume.rolling(length).sum()

    # Offset
    cmf = cmf.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        cmf.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        cmf.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    cmf.name = f"CMF_{length}"
    cmf.category = 'volume'

    return cmf


def efi(close:pd.Series, volume:pd.Series, length=None, offset=None, mamode=None, drift=None, **kwargs):
    """Elder's Force Index (EFI)
    
    Use help(df.ta.efi) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate arguments
    close = verify_series(close)
    volume = verify_series(volume)
    length = int(length) if length and length > 0 else 1
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    drift = get_drift(drift)
    offset = get_offset(offset)

    # Calculate Result
    pv_diff = close.diff(drift) * volume

    if mamode is None or mamode == 'alexander':
        efi = pv_diff.ewm(span=length, min_periods=min_periods).mean()
    else:
        efi = pv_diff.rolling(length, min_periods=min_periods).mean()

    # Offset
    efi = efi.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        efi.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        efi.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    efi.name = f"EFI_{length}"
    efi.category = 'volume'

    return efi


def eom(high:pd.Series, low:pd.Series, close:pd.Series, volume:pd.Series, length=None, divisor=None, offset=None, ease=None, **kwargs):
    """Elder's Force Index (EFI)
    
    Use help(df.ta.efi) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    volume = verify_series(volume)
    length = int(length) if length and length > 0 else 1
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    divisor = divisor if divisor and divisor > 0 else 100000000
    ease = int(ease) if ease and ease > 0 else 1
    offset = get_offset(offset)

    # Calculate Result
    hl_range = high - low
    distance = hl2(high=high, low=low) - hl2(high=high.shift(ease), low=low.shift(ease))
    box_ratio = (volume / divisor) / hl_range
    eom = distance / box_ratio
    eom = eom.rolling(length, min_periods=min_periods).mean()

    # Offset
    eom = eom.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        eom.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        eom.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    eom.name = f"EOM_{length}_{divisor}"
    eom.category = 'volume'

    return eom


def nvi(close:pd.Series, volume:pd.Series, length=None, initial=None, signed=True, offset=None, **kwargs):
    """Negative Volume Index (NVI)
    
    Use help(df.ta.nvi) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate arguments
    close = verify_series(close)
    volume = verify_series(volume)
    length = int(length) if length and length > 0 else 1
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    initial = int(initial) if initial and initial > 0 else 1000
    offset = get_offset(offset)

    # Calculate Result
    roc_ = roc(close=close)
    signed_volume = signed_series(volume, initial=1)
    nvi = signed_volume[signed_volume < 0].abs() * roc_
    nvi.fillna(0, inplace=True)
    nvi.iloc[0]= initial
    nvi = nvi.cumsum()

    # Offset
    nvi = nvi.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        nvi.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        nvi.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    nvi.name = f"NVI_{length}"
    nvi.category = 'volume'

    return nvi


def obv(close:pd.Series, volume:pd.Series, offset=None, **kwargs):
    """On Balance Volume (OBV)
    
    Use help(df.ta.obv) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate arguments
    close = verify_series(close)
    volume = verify_series(volume)
    offset = get_offset(offset)

    # Calculate Result
    signed_volume = signed_series(close, initial=1) * volume
    obv = signed_volume.cumsum()

    # Offset
    obv = obv.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        obv.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        obv.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    obv.name = f"OBV"
    obv.category = 'volume'

    return obv


# Legacy Code
def acc_dist_index(high, low, close, volume, fillna=False):
    """Accumulation/Distribution Index (ADI)

    Acting as leading indicator of price movements.

    https://en.wikipedia.org/wiki/Accumulation/distribution_index

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    clv = ((close - low) - (high - close)) / (high - low)
    clv = clv.fillna(0.0) # float division by zero
    ad = clv * volume
    ad = ad + ad.shift(1)
    if fillna:
        ad = ad.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(ad, name='adi')


def on_balance_volume(close, volume, fillna=False):
    """On-balance volume (OBV)

    It relates price and volume in the stock market. OBV is based on
    signed cumulative volume.

    https://en.wikipedia.org/wiki/On-balance_volume

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    obv = df['OBV']
    if fillna:
        obv = obv.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(obv, name='obv')


def on_balance_volume_mean(close, volume, n=10, fillna=False):
    """On-balance volume mean (OBV mean)

    It's based on signed cumulative volume.

    https://en.wikipedia.org/wiki/On-balance_volume

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    df = pd.DataFrame([close, volume]).transpose()
    df['OBV'] = 0
    c1 = close < close.shift(1)
    c2 = close > close.shift(1)
    if c1.any():
        df.loc[c1, 'OBV'] = - volume
    if c2.any():
        df.loc[c2, 'OBV'] = volume
    obv = df['OBV'].rolling(n).mean()
    if fillna:
        obv = obv.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(obv, name='obv')


def chaikin_money_flow(high, low, close, volume, n=20, fillna=False):
    """Chaikin Money Flow (CMF)

    It measures the amount of Money Flow Volume over a specific period.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    mfv = ((close - low) - (high - close)) / (high - low)
    mfv = mfv.fillna(0.0) # float division by zero
    mfv *= volume
    cmf = mfv.rolling(n).sum() / volume.rolling(n).sum()
    if fillna:
        cmf = cmf.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(cmf, name='cmf')


def force_index(close, volume, n=2, fillna=False):
    """Force Index (FI)

    It illustrates how strong the actual buying or selling pressure is. High
    positive values mean there is a strong rising trend, and low values signify
    a strong downward trend.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:force_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    fi = close.diff(n) * volume.diff(n)
    if fillna:
        fi = fi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(fi, name='fi_'+str(n))


def ease_of_movement(high, low, close, volume, n=20, fillna=False):
    """Ease of movement (EoM, EMV)

    It relate an asset's price change to its volume and is particularly useful
    for assessing the strength of a trend.

    https://en.wikipedia.org/wiki/Ease_of_movement

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    emv = (high.diff(1) + low.diff(1)) * (high - low) / (2 * volume)
    emv = emv.rolling(n).mean()
    if fillna:
        emv = emv.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(emv, name='eom_' + str(n))


def volume_price_trend(close, volume, fillna=False):
    """Volume-price trend (VPT)

    Is based on a running cumulative volume that adds or substracts a multiple
    of the percentage change in share price trend and current volume, depending
    upon the investment's upward or downward movements.

    https://en.wikipedia.org/wiki/Volume%E2%80%93price_trend

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    vpt = volume * ((close - close.shift(1)) / close.shift(1))
    vpt = vpt.shift(1) + vpt
    if fillna:
        vpt = vpt.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(vpt, name='vpt')


def negative_volume_index(close, volume, fillna=False):
    """Negative Volume Index (NVI)

    From: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde

    The Negative Volume Index (NVI) is a cumulative indicator that uses the change in volume to decide when the
    smart money is active. Paul Dysart first developed this indicator in the 1930s. [...] Dysart's Negative Volume
    Index works under the assumption that the smart money is active on days when volume decreases and the not-so-smart
    money is active on days when volume increases.

    The cumulative NVI line was unchanged when volume increased from one period to the other. In other words,
    nothing was done. Norman Fosback, of Stock Market Logic, adjusted the indicator by substituting the percentage
    price change for Net Advances.

    This implementation is the Fosback version.

    If today's volume is less than yesterday's volume then:
        nvi(t) = nvi(t-1) * ( 1 + (close(t) - close(t-1)) / close(t-1) )
    Else
        nvi(t) = nvi(t-1)

    Please note: the "stockcharts.com" example calculation just adds the percentange change of price to previous
    NVI when volumes decline; other sources indicate that the same percentage of the previous NVI value should
    be added, which is what is implemented here.

    Args:
        close(pandas.Series): dataset 'Close' column.
        volume(pandas.Series): dataset 'Volume' column.
        fillna(bool): if True, fill nan values with 1000.

    Returns:
        pandas.Series: New feature generated.

    See also:
    https://en.wikipedia.org/wiki/Negative_volume_index
    """
    price_change = close.pct_change()
    vol_decrease = (volume.shift(1) > volume)

    nvi = pd.Series(data=np.nan, index=close.index, dtype='float64', name='nvi')

    nvi.iloc[0] = 1000
    for i in range(1,len(nvi)):
        if vol_decrease.iloc[i]:
            nvi.iloc[i] = nvi.iloc[i - 1] * (1.0 + price_change.iloc[i])
        else:
            nvi.iloc[i] = nvi.iloc[i - 1]

    if fillna:
        nvi = nvi.replace([np.inf, -np.inf], np.nan).fillna(1000) # IDEA: There shouldn't be any na; might be better to throw exception

    return pd.Series(nvi, name='nvi')

# TODO

def put_call_ratio():
    # will need options volumes for this put/call ratio

    """Put/Call ratio (PCR)
    https://en.wikipedia.org/wiki/Put/call_ratio
    """
    # TODO
    return
