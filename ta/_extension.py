# -*- coding: utf-8 -*-
import time
import math
import re
import numpy as np
import pandas as pd

from .momentum import *
from .overlap import *
from .performance import *
from .statistics import *
from .trend import *
from .volatility import *
from .volume import *

from .utils import verify_series
from pandas.core.base import PandasObject



class BasePandasObject(PandasObject):
    """Simple PandasObject Extension

    Ensures the DataFrame is not empty and has columns.

    Args:
        df (pd.DataFrame): Extends Pandas DataFrame
    """
    def __init__(self, df, **kwargs):
        if df.empty: return

        total_columns = len(df.columns)
        if total_columns > 0:
            # df._total_columns = total_columns
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


    def _append(self, result=None, **kwargs):
        """Appends a Pandas Series or DataFrame columns of the result to self._df."""
        if 'append' in kwargs and kwargs['append']:
            df = self._df
            if df is None or result is None: return
            else:                
                if isinstance(result, pd.DataFrame):
                    for i, column in enumerate(result.columns):
                        df[column] = result.iloc[:,i]
                else:
                    df[result.name] = result


    def _get_column(self, series, default):
        """Attempts to get the correct series or 'column' and return it."""
        df = self._df
        if df is None: return

        # Explicit passing a pd.Series to override default.
        if isinstance(series, pd.Series):
            return series
        # Apply default if no series nor a default.
        elif series is None or default is None:
            return df[default]
        # Ok.  So it's a str.
        elif isinstance(series, str):
            # Return the df column since it's in there.
            if series in df.columns:
                return df[series]
            else:
                # Attempt to match the 'series' because it was likely misspelled.
                matches = df.columns.str.match(series, case=False)
                match = [i for i, x in enumerate(matches) if x]
                # If found, awesome.  Return it or return the 'series'.
                NOT_FOUND = f"[X] Ooops!!!: It's {series not in df.columns}, the series '{series}' not in {', '.join(list(df.columns))}"
                return df.iloc[:,match[0]] if len(match) else  print(NOT_FOUND)
        

    # Misc Methods
    def constants(self, value, min_range:int = -100, max_range:int = 100, every:int = 10):
        """Creates or removes columns of a range of integers.  Useful for indicator levels."""
        levels = [x for x in range(min_range, max_range + 1) if x % every == 0]
        if value:
            for x in levels:
                self._df[f'{x}'] = x
        else:
            for x in levels:
                del self._df[f'{x}']


    ## Momentum Indicators
    def ao(self, high=None, low=None, fast:int = None, slow:int = None, offset=None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')

        result = ao(high=high, low=low, fast=fast, slow=slow, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def apo(self, close=None, fast:int = None, slow:int = None, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = apo(close=close, fast=fast, slow=slow, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def aroon(self, close=None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = aroon(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def bop(self, open_:str = None, high:str = None, low:str = None, close:str = None, percentage:bool = False, offset=None, **kwargs):
        # Get the correct column(s).
        open_ = self._get_column(open_, 'open')
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')
        
        result = bop(open_=open_, high=high, low=low, close=close, percentage=percentage, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def cci(self, high:str = None, low:str = None, close:str = None, length:int = None, c:float = None, offset=None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result = cci(high=high, low=low, close=close, length=length, c=c, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def macd(self, close=None, fast:int = None, slow:int = None, signal:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = macd(close=close, fast=fast, slow=slow, signal=signal, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def massi(self, high:str = None, low:str = None, fast=None, slow=None, offset=None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')

        result = massi(high=high, low=low, fast=fast, slow=slow, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def mfi(self, high:str = None, low:str = None, close:str = None, volume:str = None, length:int = None, drift:int = None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        result = mfi(high=high, low=low, close=close, volume=volume, length=length, drift=drift, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def mom(self, close:str = None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = mom(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def ppo(self, close:str = None, fast:int = None, slow:int = None, percentage=True, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = ppo(close=close, fast=fast, slow=slow, percentage=percentage, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def roc(self, close:str = None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = roc(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def rsi(self, close:str = None, length:int = None, drift:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = rsi(close=close, length=length, drift=drift, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def stoch(self, high:str = None, low:str = None, close:str = None, fast_k:int = None, slow_k:int = None, slow_d:int = None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result = stoch(high=high, low=low, close=close, fast_k=fast_k, slow_k=slow_k, slow_d=slow_d, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def trix(self, close=None, length:int = None, drift:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = trix(close=close, length=length, drift=drift, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def tsi(self, close=None, fast:int = None, slow:int = None, drift:int = None, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = tsi(close=close, fast=fast, slow=slow, drift=drift, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def uo(self, high=None, low=None, close=None, fast:int = None, medium:int = None, slow:int = None, fast_w:int = None, medium_w:int = None, slow_w:int = None, drift:int = None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result = uo(high=high, low=low, close=close, fast=fast, medium=medium, slow=slow, fast_w=fast_w, medium_w=medium_w, slow_w=slow_w, drift=drift, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def willr(self, high:str = None, low:str = None, close:str = None, length:int = None, percentage:bool = True, offset:int = None,**kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')
        
        result = willr(high=high, low=low, close=close, length=length, percentage=percentage, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result



    ## Overlap Indicators
    def hl2(self, high=None, low=None, offset=None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')

        result = hl2(high=high, low=low, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def hlc3(self, high=None, low=None, close=None, offset=None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result = hlc3(high=high, low=low, close=close, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def ohlc4(self, open_=None, high=None, low=None, close=None, offset=None, **kwargs):
        # Get the correct column(s).
        open_ = self._get_column(open_, 'open')
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result = ohlc4(open_=open_, high=high, low=low, close=close, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def midpoint(self, close=None, length=None, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = midpoint(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)        
        return result


    def midprice(self, high=None, low=None, length=None, offset=None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')

        result = midprice(high=high, low=low, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)        
        return result


    def dema(self, close=None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = dema(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def ema(self, close=None, length:int = None, offset:int = None, adjust:bool = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = ema(close=close, length=length, offset=offset, adjust=adjust, **kwargs)
        self._append(result, **kwargs)
        return result


    def hma(self, close=None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = hma(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def rma(self, close=None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = rma(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def sma(self, close=None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = sma(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def t3(self, close=None, length:int = None, a:float = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = t3(close=close, length=length, a=a, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def tema(self, close=None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = tema(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def trima(self, close=None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = trima(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def vwap(self, high=None, low=None, close=None, volume=None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        result = vwap(high=high, low=low, close=close, volume=volume, offset=offset, **kwargs)
        self._append(result, **kwargs)        
        return result


    def vwma(self, close=None, volume=None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column(s).
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        result = vwma(close=close, volume=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def wma(self, close=None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = wma(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    ## Performance Indicators
    def log_return(self, close=None, length=None, cumulative:bool = False, percent:bool = False, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = log_return(close=close, length=length, cumulative=cumulative, percent=percent, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def percent_return(self, close=None, length=None, cumulative:bool = False, percent:bool = False, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = percent_return(close=close, length=length, cumulative=cumulative, percent=percent, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result



    ## Statistics Indicators
    def kurtosis(self, close=None, length=None, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = kurtosis(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def median(self, close=None, length=None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = median(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def quantile(self, close=None, length=None, q=None, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = quantile(close=close, length=length, q=q, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def skew(self, close=None, length=None, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = skew(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def stdev(self, close=None, length=None, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = stdev(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def variance(self, close=None, length=None, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = variance(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def zscore(self, close=None, length=None, std=None, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = zscore(close=close, length=length, std=std, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result



    ## Trend Indicators
    def adx(self, high=None, low=None, close=None, drift:int = None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result = adx(high=high, low=low, close=close, drift=drift, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def decreasing(self, close:str = None, length:int = None, asint:bool = True, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = decreasing(close=close, length=length, asint=asint, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def dpo(self, close:str = None, length:int = None, centered:bool = True, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = dpo(close=close, length=length, centered=centered, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def ichimoku(self, high:str = None, low:str = None, close:str = None, tenkan=None, kijun=None, senkou=None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result, span = ichimoku(high=high, low=low, close=close, tenkan=tenkan, kijun=kijun, senkou=senkou, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result, span


    def increasing(self, close:str = None, length:int = None, asint:bool = True, offset=None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = increasing(close=close, length=length, asint=asint, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def kst(self, close=None, roc1:int = None, roc2:int = None, roc3:int = None, roc4:int = None, sma1:int = None, sma2:int = None, sma3:int = None, sma4:int = None, signal:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = kst(close=close, roc1=roc1, roc2=roc2, roc3=roc3, roc4=roc4, sma1=sma1, sma2=sma2, sma3=sma3, sma4=sma4, signal=signal, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result



    ## Volatility Indicators
    def atr(self, high=None, low=None, close=None, length=None, mamode:str = None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result = atr(high=high, low=low, close=close, length=length, mamode=mamode, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def bbands(self, close=None, length:int = None, stdev:float = None, mamode:str = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')
        
        result = bbands(close=close, length=length, stdev=stdev, mamode=mamode, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def donchian(self, close=None, length:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        close = self._get_column(close, 'close')

        result = donchian(close=close, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def kc(self, high=None, low=None, close=None, length=None, scalar=None, mamode:str = None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result = kc(high=high, low=low, close=close, length=length, scalar=scalar, mamode=mamode, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def natr(self, high=None, low=None, close=None, length=None, mamode:str = None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result = natr(high=high, low=low, close=close, length=length, mamode=mamode, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def true_range(self, high=None, low=None, close=None, drift:int = None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result = true_range(high=high, low=low, close=close, drift=drift, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def vortex(self, high=None, low=None, close=None, drift:int = None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')

        result = vortex(high=high, low=low, close=close, drift=drift, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result



    ## Volume Indicators
    def ad(self, high=None, low=None, close=None, volume=None, open_=None, signed:bool = True, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        if open_ is not None:
            open_ = self._get_column(open_, 'open')

        result = ad(high=high, low=low, close=close, volume=volume, open_=open_, signed=signed, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def adosc(self, high=None, low=None, close=None, volume=None, open_=None, fast:int = None, slow:int = None, signed:bool = True, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        if open_ is not None:
            open_ = self._get_column(open_, 'open')

        result = adosc(high=high, low=low, close=close, volume=volume, open_=open_, fast=fast, slow=slow, signed=signed, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def cmf(self, high=None, low=None, close=None, volume=None, open_=None, length=None, offset:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        if open_ is not None:
            open_ = self._get_column(open_, 'open')

        result = cmf(high=high, low=low, close=close, volume=volume, open_=open_, length=length, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def efi(self, close=None, volume=None, length=None, mamode:str = None, offset:int = None, drift:int = None, **kwargs):
        # Get the correct column(s).
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        result = efi(close=close, volume=volume, length=length, offset=offset, mamode=mamode, drift=drift, **kwargs)
        self._append(result, **kwargs)
        return result


    def eom(self, high=None, low=None, close=None, volume=None, length=None, divisor:int = None, offset:int = None, drift:int = None, **kwargs):
        # Get the correct column(s).
        high = self._get_column(high, 'high')
        low = self._get_column(low, 'low')
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        result = eom(high=high, low=low, close=close, volume=volume, length=length, divisor=divisor, offset=offset, drift=drift, **kwargs)
        self._append(result, **kwargs)
        return result


    def nvi(self, close=None, volume=None, length:int = None, initial:int = None, signed:bool = True, offset:int = None, **kwargs):
        # Get the correct column(s).
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        result = nvi(close=close, volume=volume, length=length, initial=initial, signed=signed, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def obv(self, close=None, volume=None, offset:int = None, **kwargs):
        # Get the correct column(s).
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        result = obv(close=close, volume=volume, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def pvol(self, close:str = None, volume:str = None, signed:bool = True, offset:int = None, **kwargs):
        # Get the correct column(s).
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        result = pvol(close=close, volume=volume, signed=signed, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result


    def pvt(self, close=None, volume=None, offset:int = None, **kwargs):
        # Get the correct column(s).
        close = self._get_column(close, 'close')
        volume = self._get_column(volume, 'volume')

        result = pvt(close=close, volume=volume, offset=offset, **kwargs)
        self._append(result, **kwargs)
        return result



    ## Indicator Aliases by Category, more to be added later...
    # # Momentum: momomentum.py ✅
    # AbsolutePriceOscillator = apo #🤦🏻‍♂️
    # AwesomeOscillator = ao
    # BalanceOfPower = bop
    # CommodityChannelIndex = cci
    # KnowSureThing = kst
    # MACD = macd
    # MassIndex = massi
    # Momentum = mom
    # PercentagePriceOscillator = ppo
    # RateOfChange = roc
    # RelativeStrengthIndex = rsi
    # Stochastic = stoch
    # TrueStrengthIndex = tsi
    # UltimateOscillator = uo
    # WilliamsR = willr

    # # Overlap: overlap.py ✅
    # HL2 = hl2
    # HLC3 = TypicalPrice = hlc3
    # OHLC4 = ohlc4
    # Median = median
    # Midpoint = midpoint
    # Midprice = midprice
    # DoubleExponentialMovingAverage = dema
    # ExponentialMovingAverage = ema
    # HullMovingAverage = hma
    # SimpleMovingAverage = sma
    # TriangularMovingAverage = trima # require scipy
    # VolumeWeightedAveragePrice = vwap
    # VolumeWeightedMovingAverage = vwma
    # WeightedMovingAverage = wma

    # # Performance: performance.py ✅
    # LogReturn = log_return
    # PctReturn = percent_return

    # # Statistics: statistics.py ✅
    # Kurtosis = kurtosis
    # Quantile = quantile
    # Skew = skew
    # StandardDeviation = stdev
    # Variance = variance
    # ZScore = zscore

    # # Trend: trend.py ✅
    # AverageDirectionalMovmentIndex = adx
    # Decreasing = decreasing
    # DetrendPriceOscillator = dpo
    # Increasing = increasing
    # Vortex = vortex

    # # Volatility: volatility.py ✅
    # AverageTrueRange = atr
    # BollingerBands = bbands
    # DonchianChannels = donchian
    # KeltnerChannels = kc
    # NormalizedAverageTrueRange = natr
    # TrueRange = true_range

    # # Volume: volume.py ✅
    # AccumDist = ad
    # AccumDistOscillator = adosc
    # ChaikinMoneyFlow = cmf
    # EldersForceIndex = efi
    # EaseOfMovement = eom
    # NegativeVolumeIndex = nvi
    # OnBalanceVolume = obv
    # PriceVolume = pvol
    # PriceVolumeTrend = pvt


ta_indicators = list((x for x in dir(pd.DataFrame().ta) if not x.startswith('_') and not x.endswith('_')))
if True:
    print(f"[i] TA Indicators: {len(ta_indicators)}\n{', '.join(ta_indicators)}")
