# -*- coding: utf-8 -*-
"""
.. module:: momentum
   :synopsis: Momentum Indicators.

.. moduleauthor:: Dario Lopez Padial (Bukosabino)

"""
import numpy as np
import pandas as pd

from .utils import *
from .overlap import hlc3
# from .utils import get_drift, get_offset, signed_series, verify_series


def apo(close:pd.Series, fast=None, slow=None, offset=None, **kwargs):
    """Absolute Price Oscillator of a Pandas Series
    
    Use help(df.ta.apo) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    if slow < fast:
        fast, slow = slow, fast
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else fast
    offset = get_offset(offset)

    # Calculate Result
    fastma = close.rolling(fast, min_periods=min_periods).mean()
    slowma = close.rolling(slow, min_periods=min_periods).mean()
    apo = fastma - slowma

    # Handle fills
    if 'fillna' in kwargs:
        apo.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        apo.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    apo.name = f"APO_{fast}_{slow}"
    apo.category = 'momentum'

    return apo


def ao(high:pd.Series, low:pd.Series, fast=None, slow=None, offset=None, **kwargs):
    """Awesome Oscillator of a Pandas Series
    
    Use help(df.ta.ao) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    fast = int(fast) if fast and fast > 0 else 5
    slow = int(slow) if slow and slow > 0 else 34
    if slow < fast:
        fast, slow = slow, fast
    offset = get_offset(offset)

    # Calculate Result
    median_price = 0.5 * (high + low)
    ao = median_price.rolling(fast).mean() - median_price.rolling(slow).mean()

    # Offset
    ao = ao.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        ao.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        ao.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    ao.name = f"AO_{fast}_{slow}"
    ao.category = 'momentum'

    return ao


def bop(open_:pd.Series, high:pd.Series, low:pd.Series, close:pd.Series, percentage=False, offset=None, **kwargs):
    """Balance of Power of a Pandas Series
    
    Use help(df.ta.bop) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    open_ = verify_series(open_)
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    percent = 100 if percentage else 1
    offset = get_offset(offset)

    # Calculate Result
    close_open_range = close - open_
    high_log_range = high - low
    bop = percent * close_open_range / high_log_range

    # Offset
    bop = bop.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        bop.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        bop.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    bop.name = f"BOP"
    bop.category = 'momentum'

    return bop


def cci(high:pd.Series, low:pd.Series, close:pd.Series, length=None, c=None, offset=None, **kwargs):
    """Commodity Channel Index of a Pandas Series
    
    Use help(df.ta.cci) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    length = int(length) if length and length > 0 else 20
    c = float(c) if c and c > 0 else 0.015
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    def mad(series):
        """Mean Absolute Deviation"""
        return np.fabs(series - series.mean()).mean()

    typical_price = hlc3(high=high, low=low, close=close)
    mean_typical_price = typical_price.rolling(length, min_periods=min_periods).mean()
    mad_typical_price = typical_price.rolling(length).apply(mad, raw=True)

    cci = (typical_price - mean_typical_price) / (c * mad_typical_price)

    # Offset
    cci = cci.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        cci.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        cci.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    cci.name = f"CCI_{length}_{c}"
    cci.category = 'momentum'

    return cci


def macd(close:pd.Series, fast=None, slow=None, signal=None, offset=None, **kwargs):
    """Moving Average, Convergence/Divergence (MACD) of a Pandas Series
    
    Use help(df.ta.macd) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate arguments
    close = verify_series(close)
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    signal = int(signal) if signal and signal > 0 else 9
    if slow < fast:
        fast, slow = slow, fast
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else fast
    offset = get_offset(offset)

    # Calculate Result
    fastma = close.ewm(span=fast, min_periods=min_periods).mean()
    slowma = close.ewm(span=slow, min_periods=min_periods).mean()

    macd = fastma - slowma
    signalma = macd.ewm(span=signal, min_periods=min_periods).mean()
    histogram = macd - signalma

    # Offset
    macd = macd.shift(offset)
    histogram = histogram.shift(offset)
    signalma = signalma.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        macd.fillna(kwargs['fillna'], inplace=True)
        histogram.fillna(kwargs['fillna'], inplace=True)
        signalma.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        macd.fillna(method=kwargs['fill_method'], inplace=True)
        histogram.fillna(method=kwargs['fill_method'], inplace=True)
        signalma.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    macd.name = f"MACD_{fast}_{slow}_{signal}"
    histogram.name = f"MACDH_{fast}_{slow}_{signal}"
    signalma.name = f"MACDS_{fast}_{slow}_{signal}"
    macd.category = histogram.category = signalma.category = 'momentum'

    # Prepare DataFrame to return
    data = {macd.name: macd, histogram.name: histogram, signalma.name: signalma}
    macddf = pd.DataFrame(data)
    macddf.name = f"MACD_{fast}_{slow}_{signal}"
    macddf.category = 'momentum'

    return macddf


def mom(close:pd.Series, length=None, offset=None, **kwargs):
    """Momentum of a Pandas Series
    
    Use help(df.ta.mom) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 1
    offset = get_offset(offset)

    # Calculate Result
    mom = close.diff(length)

    # Offset
    mom = mom.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        mom.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        mom.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    mom.name = f"MOM_{length}"
    mom.category = 'momentum'

    return mom


def massi(high:pd.Series, low:pd.Series, fast=None, slow=None, offset=None, **kwargs):
    """Mass Index of a Pandas Series
    
    Use help(df.ta.massi) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
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
    hl_ema1 = hl_range.ewm(span=fast, min_periods=min_periods).mean()
    hl_ema2 =  hl_ema1.ewm(span=fast, min_periods=min_periods).mean()

    mass = hl_ema1 / hl_ema2
    massi = mass.rolling(slow, min_periods=slow).sum()

    # Offset
    massi = massi.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        massi.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        massi.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    # bop.name = f"BOP_{length}"
    massi.name = f"MASSI_{fast}_{slow}"
    massi.category = 'momentum'

    return massi


def mfi(high:pd.Series, low:pd.Series, close:pd.Series, volume:pd.Series, length=None, drift=None, offset=None, **kwargs):
    """Money Flow Index of a Pandas Series
    
    Use help(df.ta.mfi) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    volume = verify_series(volume)
    length = int(length) if length and length > 0 else 14
    drift = get_drift(drift)
    offset = get_offset(offset)

    # Calculate Result
    typical_price = hlc3(high=high, low=low, close=close)
    raw_money_flow = typical_price * volume

    tdf = pd.DataFrame({'diff': 0, 'rmf': raw_money_flow, '+mf': 0, '-mf': 0})

    tdf.loc[(typical_price.diff(drift) > 0), 'diff'] =  1
    tdf.loc[tdf['diff'] ==  1, '+mf'] = raw_money_flow

    tdf.loc[(typical_price.diff(drift) < 0), 'diff'] = -1
    tdf.loc[tdf['diff'] == -1, '-mf'] = raw_money_flow

    psum = tdf['+mf'].rolling(length).sum()
    nsum = tdf['-mf'].rolling(length).sum()
    tdf['mr'] = psum / nsum
    mfi = 100 * psum / (psum + nsum)
    tdf['mfi'] = mfi

    # Handle fills
    if 'fillna' in kwargs:
        mfi.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        mfi.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    mfi.name = f"MFI_{length}"
    mfi.category = 'momentum'

    return mfi


def ppo(close:pd.Series, fast=None, slow=None, percentage=True, offset=None, **kwargs):
    """Percentage Price Oscillator of a Pandas Series
    
    Use help(df.ta.ppo) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    fast = int(fast) if fast and fast > 0 else 12
    slow = int(slow) if slow and slow > 0 else 26
    if slow < fast:
        fast, slow = slow, fast
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else fast
    percent = 100 if percentage else 1
    offset = get_offset(offset)

    # Calculate Result
    fastma = close.rolling(fast, min_periods=min_periods).mean()
    slowma = close.rolling(slow, min_periods=min_periods).mean()
    ppo = percent * (fastma - slowma) / slowma

    # Handle fills
    if 'fillna' in kwargs:
        ppo.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        ppo.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    ppo.name = f"PPO_{fast}_{slow}"
    ppo.category = 'momentum'

    return ppo


def roc(close:pd.Series, length=None, offset=None, **kwargs):
    """Rate of Change (ROC) of a Pandas Series
    
    Use help(df.ta.roc) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 1
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = get_offset(offset)

    # Calculate Result
    roc = 100 * mom(close=close, length=length) / close.shift(length)

    # Offset
    roc = roc.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        roc.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        roc.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    roc.name = f"ROC_{length}"
    roc.category = 'momentum'

    return roc


def rsi(close:pd.Series, length=None, drift=None, offset=None, **kwargs):
    # Validate arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 14
    drift = get_drift(drift)
    offset = get_offset(offset)

    # Calculate Result
    negative = close.diff(drift)
    positive = negative.copy()

    positive[positive < 0] = 0  # Make negatives 0 for the postive series
    negative[negative > 0] = 0  # Make postives 0 for the negative series

    positive_avg = positive.ewm(com=length, adjust=False).mean()
    negative_avg = negative.ewm(com=length, adjust=False).mean().abs()

    rsi = 100 * positive_avg / (positive_avg + negative_avg)

    # Offset
    rsi = rsi.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        rsi.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        rsi.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    rsi.name = f"RSI_{length}"
    rsi.category = 'momentum'

    return rsi


def _stoch(df, high, low, close, fast_k:int = None, slow_k:int = None, slow_d:int = None, **kwargs):
    """Stochastic"""
    if df is None: return
    else:
        # Get the correct column.
        if isinstance(high, pd.Series):
            high = high
        else:
            high = df[high] if high in df.columns else df.high

        if isinstance(low, pd.Series):
            low = low
        else:
            low = df[low] if low in df.columns else df.low

        if isinstance(close, pd.Series):
            close = close
        else:
            close = df[close] if close in df.columns else df.close

    # Validate arguments
    fast_k = fast_k if fast_k and fast_k > 0 else 14
    slow_k = slow_k if slow_k and slow_k > 0 else 5
    slow_d = slow_d if slow_d and slow_d > 0 else 3

    # Calculate Result
    lowest_low   =  low.rolling(fast_k, min_periods=fast_k - 1).min()
    highest_high = high.rolling(fast_k, min_periods=fast_k - 1).max()

    fastk = 100 * (close - lowest_low) / (highest_high - lowest_low)
    fastd = fastk.rolling(slow_d, min_periods=slow_d - 1).mean()

    slowk = fastk.rolling(slow_k, min_periods=slow_k).mean()
    slowd = slowk.rolling(slow_d, min_periods=slow_d).mean()

    # Handle fills
    if 'fillna' in kwargs:
        fastk.fillna(kwargs['fillna'], inplace=True)
        fastd.fillna(kwargs['fillna'], inplace=True)
        slowk.fillna(kwargs['fillna'], inplace=True)
        slowd.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        fastk.fillna(method=kwargs['fill_method'], inplace=True)
        fastd.fillna(method=kwargs['fill_method'], inplace=True)
        slowk.fillna(method=kwargs['fill_method'], inplace=True)
        slowd.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    fastk.name = f"STOCHF_{fast_k}"
    fastd.name = f"STOCHF_{slow_d}"
    slowk.name = f"STOCH_{slow_k}"
    slowd.name = f"STOCH_{slow_d}"
    fastk.category = fastd.category = slowk.category = slowd.category = 'momentum'

    # If append, then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[fastk.name] = fastk
        df[fastd.name] = fastd
        df[slowk.name] = slowk
        df[slowd.name] = slowd

    # Prepare DataFrame to return
    data = {fastk.name: fastk, fastd.name: fastd, slowk.name: slowk, slowd.name: slowd}
    stochdf = pd.DataFrame(data)
    stochdf.name = f"STOCH_{fast_k}_{slow_k}_{slow_d}"
    stochdf.category = 'volatility'

    return stochdf


def tsi(close:pd.Series, fast=None, slow=None, drift=None, offset=None, **kwargs):
    """True Strength Index of a Pandas Series
    
    Use help(df.ta.tsi) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    fast = int(fast) if fast and fast > 0 else 13
    slow = int(slow) if slow and slow > 0 else 25
    if slow < fast:
        fast, slow = slow, fast
    drift = get_drift(drift)
    offset = get_offset(offset)

    # Calculate Result
    diff = close.diff(drift)

    slow_ema = diff.ewm(span=slow).mean()
    fast_slow_ema = slow_ema.ewm(span=fast).mean()

    _ma = abs(diff).ewm(span=slow).mean()
    ma = _ma.ewm(span=fast).mean()

    tsi = 100 * fast_slow_ema / ma

    # Offset
    tsi = tsi.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        tsi.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        tsi.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    tsi.name = f"TSI_{fast}_{slow}"
    tsi.category = 'momentum'

    return tsi


def uo(high:pd.Series, low:pd.Series, close:pd.Series, fast=None, medium=None, slow=None, fast_w=None, medium_w=None, slow_w=None, percentage=True, drift=None, offset=None, **kwargs):
    """Ultimate Oscillator of a Pandas Series
    
    Use help(df.ta.uo) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    percent = 100 if percentage else 1
    drift = get_drift(drift)
    offset = get_offset(offset)

    fast = int(fast) if fast and fast > 0 else 7
    fast_w = float(fast_w) if fast_w and fast_w > 0 else 4.0

    medium = int(medium) if medium and medium > 0 else 14
    medium_w = float(medium_w) if medium_w and medium_w > 0 else 2.0

    slow = int(slow) if slow and slow > 0 else 28
    slow_w = float(slow_w) if slow_w and slow_w > 0 else 1.0

    # Calculate Result
    min_l_or_pc = close.shift(drift).combine(low, min)
    max_h_or_pc = close.shift(drift).combine(high, max)

    bp = close - min_l_or_pc
    tr = max_h_or_pc - min_l_or_pc

    fast_avg = bp.rolling(fast).sum() / tr.rolling(fast).sum()
    medium_avg = bp.rolling(medium).sum() / tr.rolling(medium).sum()
    slow_avg = bp.rolling(slow).sum() / tr.rolling(slow).sum()

    total_weight =  fast_w + medium_w + slow_w
    weights = (fast_w * fast_avg) + (medium_w * medium_avg) + (slow_w * slow_avg)
    uo = percent * weights / total_weight

    # Offset
    uo = uo.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        uo.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        uo.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    uo.name = f"UO_{fast}_{medium}_{slow}"
    uo.category = 'momentum'

    return uo


def willr(high=None, low=None, close=None, length=None, percentage=True, offset=None, **kwargs):
    """William's Percent R of a Pandas Series
    
    Use help(df.ta.willr) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    length = int(length) if length and length > 0 else 14
    min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    percent = 100 if percentage else 1
    offset = get_offset(offset)

    # Calculate Result
    lowest_low = low.rolling(length, min_periods=min_periods).min()
    highest_high = high.rolling(length, min_periods=min_periods).max()

    willr = percent * ((close - lowest_low) / (highest_high - lowest_low) - 1)

    # Offset
    willr = willr.shift(offset)

    # Handle fills
    if 'fillna' in kwargs:
        willr.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        willr.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    willr.name = f"WILLR_{length}"
    willr.category = 'momentum'

    return willr



# Legacy code
def rsi_depreciated(close, n=14, fillna=False):
    """Relative Strength Index (RSI)

    Compares the magnitude of recent gains and losses over a specified time
    period to measure speed and change of price movements of a security. It is
    primarily used to attempt to identify overbought or oversold conditions in
    the trading of an asset.

    https://www.investopedia.com/terms/r/rsi.asp

    Args:
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    diff = close.diff()
    which_dn = diff < 0

    up, dn = diff, diff*0
    up[which_dn], dn[which_dn] = 0, -up[which_dn]

    emaup = ema(up, n)
    emadn = ema(dn, n)

    rsi = 100 * emaup / (emaup + emadn)
    if fillna:
        rsi = rsi.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(rsi, name='rsi')


def money_flow_index_depreciated(high, low, close, volume, n=14, fillna=False):
    """Money Flow Index (MFI)

    Uses both price and volume to measure buying and selling pressure. It is
    positive when the typical price rises (buying pressure) and negative when
    the typical price declines (selling pressure). A ratio of positive and
    negative money flow is then plugged into an RSI formula to create an
    oscillator that moves between zero and one hundred.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:money_flow_index_mfi

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
    # 0 Prepare dataframe to work
    df = pd.DataFrame([high, low, close, volume]).T
    df.columns = ['High', 'Low', 'Close', 'Volume']
    df['Up_or_Down'] = 0
    df.loc[(df['Close'] > df['Close'].shift(1)), 'Up_or_Down'] = 1
    df.loc[(df['Close'] < df['Close'].shift(1)), 'Up_or_Down'] = 2

    # 1 typical price
    tp = (df['High'] + df['Low'] + df['Close']) / 3.0

    # 2 money flow
    mf = tp * df['Volume']

    # 3 positive and negative money flow with n periods
    df['1p_Positive_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 1, '1p_Positive_Money_Flow'] = mf
    n_positive_mf = df['1p_Positive_Money_Flow'].rolling(n).sum()

    df['1p_Negative_Money_Flow'] = 0.0
    df.loc[df['Up_or_Down'] == 2, '1p_Negative_Money_Flow'] = mf
    n_negative_mf = df['1p_Negative_Money_Flow'].rolling(n).sum()

    # 4 money flow index
    mr = n_positive_mf / n_negative_mf
    mr = (100 - (100 / (1 + mr)))
    if fillna:
        mr = mr.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(mr, name='mfi_'+str(n))


def tsi_depreciated(close, r=25, s=13, fillna=False):
    """True strength index (TSI)

    Shows both trend direction and overbought/oversold conditions.

    https://en.wikipedia.org/wiki/True_strength_index

    Args:
        close(pandas.Series): dataset 'Close' column.
        r(int): high period.
        s(int): low period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    m = close - close.shift(1)
    m1 = m.ewm(r).mean().ewm(s).mean()
    m2 = abs(m).ewm(r).mean().ewm(s).mean()
    tsi = m1 / m2
    tsi *= 100
    if fillna:
        tsi = tsi.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(tsi, name='tsi')

def uo_depreciated(high, low, close, s=7, m=14, l=28, ws=4.0, wm=2.0, wl=1.0, fillna=False):
    """Ultimate Oscillator

    Larry Williams' (1976) signal, a momentum oscillator designed to capture momentum
    across three different timeframes.

    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ultimate_oscillator

    BP = Close - Minimum(Low or Prior Close).
    TR = Maximum(High or Prior Close)  -  Minimum(Low or Prior Close)
    Average7 = (7-period BP Sum) / (7-period TR Sum)
    Average14 = (14-period BP Sum) / (14-period TR Sum)
    Average28 = (28-period BP Sum) / (28-period TR Sum)

    UO = 100 x [(4 x Average7)+(2 x Average14)+Average28]/(4+2+1)

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        s(int): short period
        m(int): medium period
        l(int): long period
        ws(float): weight of short BP average for UO
        wm(float): weight of medium BP average for UO
        wl(float): weight of long BP average for UO
        fillna(bool): if True, fill nan values with 50.

    Returns:
        pandas.Series: New feature generated.

    """
    min_l_or_pc = close.shift(1).combine(low, min)
    max_h_or_pc = close.shift(1).combine(high, max)

    bp = close - min_l_or_pc
    tr = max_h_or_pc - min_l_or_pc

    avg_s = bp.rolling(s).sum() / tr.rolling(s).sum()
    avg_m = bp.rolling(m).sum() / tr.rolling(m).sum()
    avg_l = bp.rolling(l).sum() / tr.rolling(l).sum()

    uo = 100.0 * ((ws * avg_s) + (wm * avg_m) + (wl * avg_l)) / (ws + wm + wl)
    if fillna:
        uo = uo.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(uo, name='uo')


def stoch_depreciated(high, low, close, n=14, fillna=False):
    """Stochastic Oscillator

    Developed in the late 1950s by George Lane. The stochastic
    oscillator presents the location of the closing price of a
    stock in relation to the high and low range of the price
    of a stock over a period of time, typically a 14-day period.

    https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    smin = low.rolling(n).min()
    smax = high.rolling(n).max()
    stoch_k = 100 * (close - smin) / (smax - smin)

    if fillna:
        stoch_k = stoch_k.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(stoch_k, name='stoch_k')


def stoch_signal_depreciated(high, low, close, n=14, d_n=3, fillna=False):
    """Stochastic Oscillator Signal

    Shows SMA of Stochastic Oscillator. Typically a 3 day SMA.

    https://www.investopedia.com/terms/s/stochasticoscillator.asp

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        n(int): n period.
        d_n(int): sma period over stoch_k
        fillna(bool): if True, fill nan values.

    Returns:
        pandas.Series: New feature generated.
    """
    stoch_k = stoch(high, low, close, n, fillna=fillna)
    stoch_d = stoch_k.rolling(d_n).mean()

    if fillna:
        stoch_d = stoch_d.replace([np.inf, -np.inf], np.nan).fillna(50)
    return pd.Series(stoch_d, name='stoch_d')


def wr_depreciated(high, low, close, lbp=14, fillna=False):
    """Williams %R

    From: http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:williams_r

    Developed by Larry Williams, Williams %R is a momentum indicator that is the inverse of the
    Fast Stochastic Oscillator. Also referred to as %R, Williams %R reflects the level of the close
    relative to the highest high for the look-back period. In contrast, the Stochastic Oscillator
    reflects the level of the close relative to the lowest low. %R corrects for the inversion by
    multiplying the raw value by -100. As a result, the Fast Stochastic Oscillator and Williams %R
    produce the exact same lines, only the scaling is different. Williams %R oscillates from 0 to -100.

    Readings from 0 to -20 are considered overbought. Readings from -80 to -100 are considered oversold.

    Unsurprisingly, signals derived from the Stochastic Oscillator are also applicable to Williams %R.


    %R = (Highest High - Close)/(Highest High - Lowest Low) * -100

    Lowest Low = lowest low for the look-back period
    Highest High = highest high for the look-back period
    %R is multiplied by -100 correct the inversion and move the decimal.

    From: https://www.investopedia.com/terms/w/williamsr.asp
    The Williams %R oscillates from 0 to -100. When the indicator produces readings from 0 to -20, this indicates
    overbought market conditions. When readings are -80 to -100, it indicates oversold market conditions.


    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        close(pandas.Series): dataset 'Close' column.
        lbp(int): lookback period
        fillna(bool): if True, fill nan values with -50.

    Returns:
        pandas.Series: New feature generated.
    """

    hh = high.rolling(lbp).max() #highest high over lookback period lbp
    ll = low.rolling(lbp).min()  #lowest low over lookback period lbp

    wr = -100 * (hh - close) / (hh - ll)

    if fillna:
        wr = wr.replace([np.inf, -np.inf], np.nan).fillna(-50)
    return pd.Series(wr, name='wr')


def ao_depreciated(high, low, s=5, l=34, fillna=False):
    """Awesome Oscillator

    From: https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)

    The Awesome Oscillator is an indicator used to measure market momentum. AO calculates the difference of a
    34 Period and 5 Period Simple Moving Averages. The Simple Moving Averages that are used are not calculated
    using closing price but rather each bar's midpoints. AO is generally used to affirm trends or to anticipate
    possible reversals.

    From: https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator

    Awesome Oscillator is a 34-period simple moving average, plotted through the central points of the bars (H+L)/2,
    and subtracted from the 5-period simple moving average, graphed across the central points of the bars (H+L)/2.
    MEDIAN PRICE = (HIGH+LOW)/2
    AO = SMA(MEDIAN PRICE, 5)-SMA(MEDIAN PRICE, 34)

    where

    SMA — Simple Moving Average.

    Args:
        high(pandas.Series): dataset 'High' column.
        low(pandas.Series): dataset 'Low' column.
        s(int): short period
        l(int): long period
        fillna(bool): if True, fill nan values with -50.

    Returns:
        pandas.Series: New feature generated.
    """

    mp = 0.5 * (high + low)
    ao = mp.rolling(s).mean() - mp.rolling(l).mean()

    if fillna:
        ao = ao.replace([np.inf, -np.inf], np.nan).fillna(0)
    return pd.Series(ao, name='ao')
