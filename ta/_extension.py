# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import pandas as pd

from .momentum import *
# from .others import *
from .overlap import *
from .performance import *
from .statistics import *
from .trend import *
from .utils import signed_series
# from .volatility import *
from .volume import *

from pandas.core.base import PandasObject
from sys import float_info as sflt

TA_EPSILON = sflt.epsilon



def _wma(df, length:int = None, asc:bool = True, **kwargs):
    length = length if length and length > 0 else 1
    total_weight = 0.5 * length * (length + 1)
    weights_ = pd.Series(np.arange(1, length + 1))
    weights = weights_ if asc else weights_[::-1]

    def linear_weights(w):
        def _compute(x):
            return (w * x).sum() / total_weight
        return _compute

    return df.rolling(length, min_periods=length).apply(linear_weights(weights), raw=True)        

def _kst(df, close=None, roc1:int = None, roc2:int = None, roc3:int = None, roc4:int = None, sma1:int = None, sma2:int = None, sma3:int = None, sma4:int = None, signal:int = None, drift:int = None, **kwargs):
    """Know Sure Thing - kst"""
    if df is None: return
    else:
        # Get the correct column.
        if isinstance(close, pd.Series):
            close = close
        else:
            close = df[close] if close in df.columns else df.close

    # Validate arguments
    roc1 = int(roc1) if roc1 and roc1 > 0 else 10
    roc2 = int(roc2) if roc2 and roc2 > 0 else 15
    roc3 = int(roc3) if roc3 and roc3 > 0 else 20
    roc4 = int(roc4) if roc4 and roc4 > 0 else 30

    sma1 = int(sma1) if sma1 and sma1 > 0 else 10
    sma2 = int(sma2) if sma2 and sma2 > 0 else 10
    sma3 = int(sma3) if sma3 and sma3 > 0 else 10
    sma4 = int(sma4) if sma4 and sma4 > 0 else 15

    signal = int(signal) if signal and signal > 0 else 9

    # Calculate Result
    rocma1 = (close.diff(roc1) / close.shift(roc1)).rolling(sma1).mean()
    rocma2 = (close.diff(roc2) / close.shift(roc2)).rolling(sma2).mean()
    rocma3 = (close.diff(roc3) / close.shift(roc3)).rolling(sma3).mean()
    rocma4 = (close.diff(roc4) / close.shift(roc4)).rolling(sma4).mean()

    kst = 100 * (rocma1 + 2 * rocma2 + 3 * rocma3 + 4 * rocma4)
    kst_signal = kst.rolling(signal).mean()

    # Handle fills
    if 'fillna' in kwargs:
        kst.fillna(kwargs['fillna'], inplace=True)
        kst_signal.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        kst.fillna(method=kwargs['fill_method'], inplace=True)
        kst_signal.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    kst.name = f"KST_{roc1}_{roc2}_{roc3}_{roc4}_{sma1}_{sma2}_{sma3}_{sma4}"
    kst_signal.name = f"KSTS_{signal}"
    kst.category = kst_signal.category = 'momentum'

    # If append, then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[kst.name] = kst
        df[kst_signal.name] = kst_signal

    # Prepare DataFrame to return
    data = {kst.name: kst, kst_signal.name: kst_signal}
    kstdf = pd.DataFrame(data)
    kstdf.name = f"KST_{roc1}_{roc2}_{roc3}_{roc4}_{sma1}_{sma2}_{sma3}_{sma4}_{signal}"
    kstdf.category = 'momentum'

    return kstdf


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


def _uo(high=None, low=None, close=None, fast:int = None, medium:int = None, slow:int = None, fast_w:int = None, medium_w:int = None, slow_w:int = None, drift:int = None, **kwargs):
    """Ultimate Oscillator - uso"""
    # Validate arguments
    fast = int(fast) if fast and fast > 0 else 7
    fast_w = float(fast_w) if fast_w and fast_w > 0 else 4.0

    medium = int(medium) if medium and medium > 0 else 14
    medium_w = float(medium_w) if medium_w and medium_w > 0 else 2.0

    slow = int(slow) if slow and slow > 0 else 28
    slow_w = float(slow_w) if slow_w and slow_w > 0 else 1.0

    drift = int(drift) if drift and drift > 0 else 1
    # min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else fast

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
    uo = 100 * weights / total_weight
    
    # Handle fills
    if 'fillna' in kwargs:
        uo.fillna(kwargs['fillna'], inplace=True)
    if 'fill_method' in kwargs:
        uo.fillna(method=kwargs['fill_method'], inplace=True)

    # Name and Categorize it
    uo.name = f"UO_{fast}_{medium}_{slow}"
    uo.category = 'momentum'

    # If append, then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[uo.name] = uo

    return uo


class BasePandasObject(PandasObject):
    """Simple PandasObject Extension

    Ensures the DataFrame is not empty and has columns.

    Args:
        df (pd.DataFrame): Extends Pandas DataFrame
    """
    def __init__(self, df, **kwargs):
        if df.empty:
            return None

        total_columns = len(df.columns)
        if total_columns > 0:
            self._df = df
        else:
            raise AttributeError(f"[X] No columns!")

    def __call__(self, kind, *args, **kwargs):
        raise NotImplementedError()


@pd.api.extensions.register_dataframe_accessor('ta')
class AnalysisIndicators(BasePandasObject):
    """AnalysisIndicators is class that extends a DataFrame.  The name extension is
    registered to all instances of the DataFrame wit the name of 'ta'.

    Args:
        kind(str): Name of the indicator.  Converts kind to lowercase.
        kwargs(dict): Method specific modifiers

    Returns:
        Either a Pandas Series or DataFrame of the results of the called indicator.

    Example A:  Loading Data and multiply ways of calling a function.
    # Load some data. If local, this would do.
    # Assum having a CSV with columns: date,open,high,low,close,volume
    df = pd.read_csv('AAPL.csv', index_col='date', parse_dates=True, dtype=float, infer_datetime_format=False, keep_date_col=True)

    # Calling HL2.  All equivalent.  Thy return a new Series/DataFrame with
    #  the Indicator result
    hl2 = df.ta.hl2()
    hl2 = df.ta.HL2()
    hl2 = df.ta(kind='hl2')

    #Given a TimeSeries DataFrame called df with lower case column names. ie. open, high, lose, close, volume

    Additional kwargs:
    * append: Default: False.  If True, appends the indicator result to the df.
    """

    def __call__(self, kind=None, alias=None, timed=False, **kwargs):
        try:
            kind = kind.lower() if isinstance(kind, str) else None
            fn = getattr(self, kind.lower())
        except AttributeError:
            raise ValueError(f"kind='{kind.lower()}' is not valid for {self.__class__.__name__}")

        if timed:
            stime = time.time()

        # Run the indicator
        indicator = fn(**kwargs)

        if timed:
            time_diff = time.time() - stime
            ms = time_diff * 1000
            indicator.timed = f"{ms:2.3f} ms ({time_diff:2.3f} s)"
        
        # Add an alias if passed
        if alias: indicator.alias = f"{alias}"

        return indicator


    # @property
    def defaults(self, value, min_range:int = -100, max_range:int = 100, every:int = 10):
        _levels = [x for x in range(min_range, max_range + 1) if x % every == 0]
        if value:
            for x in _levels:
                self._df[f'{x}'] = x
        else:
            for x in _levels:
                del self._df[f'{x}']



    ## Momentum Indicators
    def apo(self, close=None, fast:int = None, slow:int = None, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = apo(close=close, fast=fast, slow=slow, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def ao(self, high=None, low=None, fast:int = None, slow:int = None, offset=None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(high, pd.Series):
                high = high
            else:
                high = df[high] if high in df.columns else df.high

            if isinstance(low, pd.Series):
                low = low
            else:
                low = df[low] if low in df.columns else df.low

        result = ao(high=high, low=low, fast=fast, slow=slow, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def aroon(self, close=None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = aroon(close=close, length=length, offset=offset, **kwargs)

        # # If append, then add it to the df
        # if 'append' in kwargs and kwargs['append']:
        #     df[result.name] = result

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.columns[0].name] = result.columns[0]  # aroon_up
            df[result.columns[1].name] = result.columns[1]  # aroon_down

        return result


    def bop(self, open_:str = None, high:str = None, low:str = None, close:str = None, percentage:bool = False, offset=None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(open_, pd.Series):
                open_ = open_
            else:
                open_ = df[open_] if open_ in df.columns else df.open

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
        
        result = bop(open_=open_, high=high, low=low, close=close, percentage=percentage, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def cci(self, high:str = None, low:str = None, close:str = None, length:int = None, c:float = None, offset=None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
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

        result = cci(high=high, low=low, close=close, length=length, c=c, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def kst(self, close=None, roc1:int = None, roc2:int = None, roc3:int = None, roc4:int = None, sma1:int = None, sma2:int = None, sma3:int = None, sma4:int = None, signal:int = None, **kwargs):
        result = _kst(self._df, close=close, roc1=roc1, roc2=roc2, roc3=roc3, roc4=roc4, sma1=sma1, sma2=sma2, sma3=sma3, sma4=sma4, signal=signal, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def macd(self, close=None, fast:int = None, slow:int = None, signal:int = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        fast = int(fast) if fast and fast > 0 else 12
        slow = int(slow) if slow and slow > 0 else 26
        signal = int(signal) if signal and signal > 0 else 9
        if slow < fast:
            fast, slow = slow, fast
        min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else fast

        # Calculate Result
        fastma = close.ewm(span=fast, min_periods=min_periods).mean()
        slowma = close.ewm(span=slow, min_periods=min_periods).mean()
        macd = fastma - slowma

        signalma = macd.ewm(span=signal, min_periods=min_periods).mean()
        histogram = macd - signalma

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

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[macd.name] = macd
            df[histogram.name] = histogram
            df[signalma.name] = signalma

        # Prepare DataFrame to return
        data = {macd.name: macd, histogram.name: histogram, signalma.name: signalma}
        macddf = pd.DataFrame(data)
        macddf.name = f"MACD_{fast}_{slow}_{signal}"
        macddf.category = 'momentum'

        return macddf


    def massi(self, high:str = None, low:str = None, fast=None, slow=None, offset=None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(high, pd.Series):
                high = high
            else:
                high = df[high] if high in df.columns else df.high

            if isinstance(low, pd.Series):
                low = low
            else:
                low = df[low] if low in df.columns else df.low

        result = massi(high=high, low=low, fast=fast, slow=slow, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def mfi(self, high:str = None, low:str = None, close:str = None, volume:str = None, length:int = None, drift:int = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
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

            if isinstance(volume, pd.Series):
                volume = volume
            else:
                volume = df[volume] if volume in df.columns else df.volume

        # Validate arguments
        length = int(length) if length and length > 0 else 14
        drift = int(drift) if drift and drift > 0 else 1

        # Calculate Result
        typical_price = self.hlc3(high=high, low=low, close=close)
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

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[mfi.name] = mfi

        return mfi


    def mom(self, close:str = None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = mom(close=close, length=length, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def ppo(self, close:str = None, fast:int = None, slow:int = None, percentage=True, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = ppo(close=close, fast=fast, slow=slow, percentage=percentage, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result



    def roc(self, close:str = None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = roc(close=close, length=length, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def rsi(self, close:str = None, length:int = None, drift:int = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        length = int(length) if length and length > 0 else 14
        drift = int(drift) if drift and drift > 0 else 1

        # Calculate Result
        negative = close.diff(drift)
        positive = negative.copy()

        positive[positive < 0] = 0  # Make negatives 0 for the postive series
        negative[negative > 0] = 0  # Make postives 0 for the negative series

        positive_avg = positive.ewm(com=length, adjust=False).mean()
        negative_avg = negative.ewm(com=length, adjust=False).mean().abs()

        rsi = 100 * positive_avg / (positive_avg + negative_avg)

        # Handle fills
        if 'fillna' in kwargs:
            rsi.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            rsi.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        rsi.name = f"RSI_{length}"
        rsi.category = 'momentum'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[rsi.name] = rsi

        return rsi


    def tsi(self, close=None, fast:int = None, slow:int = None, drift:int = None, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = tsi(close=close, fast=fast, slow=slow, drift=drift, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def uo(self, high=None, low=None, close=None, fast:int = None, medium:int = None, slow:int = None, fast_w:int = None, medium_w:int = None, slow_w:int = None, drift:int = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
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

        result = _uo(high=high, low=low, close=close, fast=fast, medium=medium, slow=slow, fast_w=fast_w, medium_w=medium_w, slow_w=slow_w, drift=drift, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def willr(self, high:str = None, low:str = None, close:str = None, length:int = None, percentage:bool = True, offset:int = None,**kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
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
        
        result = willr(high=high, low=low, close=close, length=length, percentage=percentage, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    ## Overlap Indicators
    def hl2(self, high=None, low=None, offset=None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(high, pd.Series):
                high = high
            else:
                high = df[high] if high in df.columns else df.high

            if isinstance(low, pd.Series):
                low = low
            else:
                low = df[low] if low in df.columns else df.low

        result = hl2(high=high, low=low, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def hlc3(self, high=None, low=None, close=None, offset=None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
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

        result = hlc3(high=high, low=low, close=close, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def ohlc4(self, open_=None, high=None, low=None, close=None, offset=None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(open_, pd.Series):
                open_ = open_
            else:
                open_ = df[open_] if open_ in df.columns else df.open

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

        result = ohlc4(open_=open_, high=high, low=low, close=close, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def median(self, close=None, length=None, offset:int = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = median(close=close, length=length, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result
 


    def midpoint(self, close=None, length=None, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = midpoint(close=close, length=length, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def midprice(self, high=None, low=None, length=None, offset=None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(low, pd.Series):
                low = low
            else:
                low = df[low] if low in df.columns else df.low

            if isinstance(high, pd.Series):
                high = high
            else:
                high = df[high] if high in df.columns else df.high

        result = midprice(high=high, low=low, length=length, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def rpn(self, high=None, low=None, length=None, offset=None, percentage=None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(high, pd.Series):
                high = high
            else:
                high = df[high] if high in df.columns else df.high

            if isinstance(low, pd.Series):
                low = low
            else:
                low = df[low] if low in df.columns else df.low

        result = rpn(high=high, low=low, length=length, offset=offset, percentage=percentage, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    # def wma(self, close:str = None, length:int = None, asc:bool = True, **kwargs):
    #     """ wma """
    #     df = self._df
        
    #     length = length if length and length > 0 else 1

    #     # Get the correct column.
        # if df is None: return
        # else:
        #     if isinstance(close, pd.Series):
        #         close = close
        #     else:
        #         close = df[close] if close in df.columns else df.close
        
    #     wma = _wma(close, length=length, **kwargs)
        
    #     # Handle fills
    #     if 'fillna' in kwargs:
    #         wma.fillna(kwargs['fillna'], inplace=True)
    #     elif 'fill_method' in kwargs:
    #         wma.fillna(method=kwargs['fill_method'], inplace=True)                

    #     # Name and Categorize it
    #     wma.name = f"WMA_{length}"
    #     wma.category = 'overlap'
        
    #     # If append, then add it to the df 
    #     if 'append' in kwargs and kwargs['append']:
    #         df[wma.name] = wma
            
    #     return wma



    ## Performance Indicators
    def log_return(self, close=None, length=None, cumulative:bool = False, percent:bool = False, offset:int = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = log_return(close=close, length=length, cumulative=cumulative, percent=percent, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def percent_return(self, close=None, length=None, cumulative:bool = False, percent:bool = False, offset:int = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = percent_return(close=close, length=length, cumulative=cumulative, percent=percent, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result



    ## Statistics Indicators
    def kurtosis(self, close=None, length=None, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = kurtosis(close=close, length=length, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def quantile(self, close=None, length=None, q=None, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close
        
        result = quantile(close=close, length=length, q=q, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def skew(self, close=None, length=None, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = skew(close=close, length=length, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def stdev(self, close=None, length=None, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = stdev(close=close, length=length, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def variance(self, close=None, length=None, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = variance(close=close, length=length, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result



    ## Trend Indicators
    def decreasing(self, close:str = None, length:int = None, asint:bool = True, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = decreasing(close=close, length=length, asint=asint, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def dpo(self, close:str = None, length:int = None, centered:bool = True, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = dpo(close=close, length=length, centered=centered, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result


    def increasing(self, close:str = None, length:int = None, asint:bool = True, offset=None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = increasing(close=close, length=length, asint=asint, offset=offset, **kwargs)

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result
        
        return result



    ## Volatility Indicators
    def atr(self, high=None, low=None, close=None, length=None, mamode:str = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
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
        length = int(length) if length and length > 0 else 14
        min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
        mamode = mamode.lower() if mamode else 'ema'

        # Calculate Result
        true_range = self.true_range(high=high, low=low, close=close, length=length)
        if mamode == 'ema':
            atr = true_range.ewm(span=length, min_periods=min_periods).mean()
        else:
            atr = true_range.rolling(length, min_periods=min_periods).mean()

        # Handle fills
        if 'fillna' in kwargs:
            atr.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            atr.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        atr.name = f"ATR_{length}"
        atr.category = 'volatility'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[atr.name] = atr

        return atr


    def bbands(self, close=None, length:int = None, stdev:float = None, mamode:str = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        length = int(length) if length and length > 0 else 20
        min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
        stdev = int(stdev) if stdev and stdev >= 0 else 2

        # Calculate Result
        std = self.variance(close=close, length=length).apply(np.sqrt)

        if mamode is None or mamode.lower() == 'sma':
            mid = close.rolling(length, min_periods=min_periods).mean()
        elif mamode.lower() == 'ema':
            mid = close.ewm(span=length, min_periods=min_periods).mean()

        lower = mid - stdev * std
        upper = mid + stdev * std

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

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[lower.name] = lower
            df[mid.name] = mid
            df[upper.name] = upper

        # Prepare DataFrame to return
        data = {lower.name: lower, mid.name: mid, upper.name: upper}
        bbandsdf = pd.DataFrame(data)
        bbandsdf.name = f"BBANDS{length}"
        bbandsdf.category = 'volatility'

        return bbandsdf


    def donchian(self, close=None, length:int = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        length = int(length) if length and length > 0 else 20
        min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length

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

        # Name and Categorize it
        lower.name = f"DCL_{length}"
        mid.name = f"DCM_{length}"
        upper.name = f"DCU_{length}"
        mid.category = upper.category = lower.category = 'volatility'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[lower.name] = lower
            df[mid.name] = mid
            df[upper.name] = upper

        # Prepare DataFrame to return
        data = {lower.name: lower, mid.name: mid, upper.name: upper}
        dcdf = pd.DataFrame(data)
        dcdf.name = f"DC{length}"
        dcdf.category = 'volatility'

        return dcdf


    def kc(self, high=None, low=None, close=None, length=None, scalar=None, mamode:str = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
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
        length = int(length) if length and length > 0 else 9999
        min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
        scalar = float(scalar) if scalar and scalar >= 0 else 2
        mamode = mamode.lower() if mamode else 'classic'

        # Calculate Result
        std = self.variance(close=close, length=length).apply(np.sqrt)

        if mamode == 'ema':
            hl_range = high - low
            typical_price = self.hlc3(high=high, low=low, close=close)
            basis = typical_price.rolling(length, min_periods=min_periods).mean()
            band = hl_range.rolling(length, min_periods=min_periods).mean()
        else:
            basis = close.ewm(span=length, min_periods=min_periods).mean()
            band = self.atr(high=high, low=low, close=close)

        lower = basis - scalar * band
        upper = basis + scalar * band

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

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[lower.name] = lower
            df[basis.name] = basis
            df[upper.name] = upper

        # Prepare DataFrame to return
        data = {lower.name: lower, basis.name: basis, upper.name: upper}
        kcdf = pd.DataFrame(data)
        kcdf.name = f"KC{length}"
        kcdf.category = 'volatility'

        return kcdf


    def stoch(self, high:str = None, low:str = None, close:str = None, fast_k:int = None, slow_k:int = None, slow_d:int = None, **kwargs):
        return _stoch(self._df, high=high, low=low, close=close, fast_k=fast_k, slow_k=slow_k, slow_d=slow_d, **kwargs)


    def true_range(self, high=None, low=None, close=None, length=None, drift:int = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
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
        length = int(length) if length and length > 0 else 1
        drift = int(drift) if drift and drift != 0 else 1
        min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length

        # Calculate Result
        prev_close = close.shift(drift)
        ranges = [high - low, high - prev_close, low - prev_close]
        true_range = pd.DataFrame(ranges).T
        true_range = true_range.abs().max(axis=1)

        # Handle fills
        if 'fillna' in kwargs:
            true_range.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            true_range.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        true_range.name = f"TRUERANGE_{length}"
        true_range.category = 'volatility'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[true_range.name] = true_range

        return true_range



    ## Volume Indicators
    def ad(self, high=None, low=None, close=None, volume=None, open_=None, signed:bool = True, offset:int = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
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

            if isinstance(volume, pd.Series):
                volume = volume
            else:
                volume = df[volume] if volume in df.columns else df.volume

            if open_ is not None:
                if isinstance(open_, pd.Series):
                    open_ = open_
                else:
                    open_ = df[open_] if open_ in df.columns else df.open

        result = ad(high=high, low=low, close=close, volume=volume, open_=open_, signed=signed, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def cmf(self, high=None, low=None, close=None, volume=None, open_=None, length=None, offset:int = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
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

            if isinstance(volume, pd.Series):
                volume = volume
            else:
                volume = df[volume] if volume in df.columns else df.volume

            if open_ is not None:
                if isinstance(open_, pd.Series):
                    open_ = open_
                else:
                    open_ = df[open_] if open_ in df.columns else df.open

        result = cmf(high=high, low=low, close=close, volume=volume, open_=open_, length=length, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def efi(self, close=None, volume=None, length=None, mamode:str = None, offset:int = None, drift:int = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

            if isinstance(volume, pd.Series):
                volume = volume
            else:
                volume = df[volume] if volume in df.columns else df.volume

        result = efi(close=close, volume=volume, length=length, offset=offset, mamode=mamode, drift=drift, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def eom(self, high=None, low=None, close=None, volume=None, length=None, divisor:int = None, offset:int = None, drift:int = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
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

            if isinstance(volume, pd.Series):
                volume = volume
            else:
                volume = df[volume] if volume in df.columns else df.volume

        result = eom(high=high, low=low, close=close, volume=volume, length=length, divisor=divisor, offset=offset, drift=drift, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def nvi(self, close=None, volume=None, length:int = None, initial:int = None, signed:bool = True, offset:int = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

            if isinstance(volume, pd.Series):
                volume = volume
            else:
                volume = df[volume] if volume in df.columns else df.volume

        result = nvi(close=close, volume=volume, length=length, initial=initial, signed=signed, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def obv(self, close=None, volume=None, offset:int = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

            if isinstance(volume, pd.Series):
                volume = volume
            else:
                volume = df[volume] if volume in df.columns else df.volume

        result = obv(close=close, volume=volume, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def pvol(self, close:str = None, volume:str = None, signed:bool = True, offset:int = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

            if isinstance(volume, pd.Series):
                volume = volume
            else:
                volume = df[volume] if volume in df.columns else df.volume

        result = pvol(close=close, volume=volume, signed=signed, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result


    def pvt(self, close=None, volume=None, offset:int = None, **kwargs):
        # Get the correct column(s).
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

            if isinstance(volume, pd.Series):
                volume = volume
            else:
                volume = df[volume] if volume in df.columns else df.volume

        result = pvt(close=close, volume=volume, offset=offset, **kwargs)

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[result.name] = result

        return result



    ## Indicator Aliases by Category
    # Momentum: momomentum.py #⏸
    AbsolutePriceOscillator = apo #🤦🏻‍♂️
    AwesomeOscillator = ao
    BalanceOfPower = bop
    CommodityChannelIndex = cci
    KnowSureThing = kst #⏸
    MACD = macd #⏸
    MassIndex = massi #🤦🏻‍♂️
    Momentum = mom
    PercentagePriceOscillator = ppo
    RateOfChange = roc
    RelativeStrengthIndex = rsi #⏸
    TrueStrengthIndex = tsi
    UltimateOscillator = uo #⏸
    WilliamsR = willr #⏸

    # Overlap: overlap.py ✅
    HL2 = hl2
    HLC3 = TypicalPrice = hlc3
    OHLC4 = ohlc4
    Median = median
    Midpoint = midpoint
    Midprice = midprice
    RangePercentage = rpn

    # Performance: performance.py ✅
    LogReturn = log_return
    PctReturn = percent_return

    # Statistics: statistics.py ✅
    Kurtosis = kurtosis
    Quantile = quantile
    Skew = skew
    StandardDeviation = stdev
    Variance = variance

    # Trend: trend.py ✅
    Decreasing = decreasing
    DetrendPriceOscillator = dpo
    Increasing = increasing

    # Volatility: volatility.py #⏸
    AverageTrueRange = atr
    BollingerBands = bbands
    DonchianChannels = donchian
    KeltnerChannels = kc
    TrueRange = true_range

    # Volume: volume.py ✅
    AccumDist = ad
    ChaikinMoneyFlow = cmf
    EldersForceIndex = efi
    EaseOfMovement = eom
    NegativeVolumeIndex = nvi
    OnBalanceVolume = obv
    PriceVolume = pvol
    PriceVolumeTrend = pvt


ta_indicators = list((x for x in dir(pd.DataFrame().ta) if not x.startswith('_') and not x.endswith('_')))
if True:
    print(f"[i] Loaded {len(ta_indicators)} TA Indicators:\n{', '.join(ta_indicators)}")
