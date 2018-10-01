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
# from sys import float_info as sflt

# TA_EPSILON = sflt.epsilon


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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

        return result


    def mfi(self, high:str = None, low:str = None, close:str = None, volume:str = None, length:int = None, drift:int = None, offset:int = None, **kwargs):
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

        result = mfi(high=high, low=low, close=close, volume=volume, length=length, drift=drift, offset=offset, **kwargs)

        self._append(result, **kwargs)

        return result


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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

        return result


    def rsi(self, close:str = None, length:int = None, drift:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = rsi(close=close, length=length, drift=drift, offset=offset, **kwargs)

        self._append(result, **kwargs)

        return result


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

        self._append(result, **kwargs)

        return result


    def uo(self, high=None, low=None, close=None, fast:int = None, medium:int = None, slow:int = None, fast_w:int = None, medium_w:int = None, slow_w:int = None, drift:int = None, offset:int = None, **kwargs):
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

        result = uo(high=high, low=low, close=close, fast=fast, medium=medium, slow=slow, fast_w=fast_w, medium_w=medium_w, slow_w=slow_w, drift=drift, offset=offset, **kwargs)

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
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

        self._append(result, **kwargs)
        
        return result


    def kst(self, close=None, roc1:int = None, roc2:int = None, roc3:int = None, roc4:int = None, sma1:int = None, sma2:int = None, sma3:int = None, sma4:int = None, signal:int = None, offset:int = None, **kwargs):
        # Get the correct column.
        df = self._df
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        result = kst(close=close, roc1=roc1, roc2=roc2, roc3=roc3, roc4=roc4, sma1=sma1, sma2=sma2, sma3=sma3, sma4=sma4, signal=signal, offset=offset, **kwargs)

        self._append(result, **kwargs)

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
        result =  _stoch(self._df, high=high, low=low, close=close, fast_k=fast_k, slow_k=slow_k, slow_d=slow_d, **kwargs)

        # self._append(result, **kwargs)
        return result


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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

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

        self._append(result, **kwargs)

        return result



    ## Indicator Aliases by Category
    # Momentum: momomentum.py #⏸
    AbsolutePriceOscillator = apo #🤦🏻‍♂️
    AwesomeOscillator = ao
    BalanceOfPower = bop
    CommodityChannelIndex = cci
    KnowSureThing = kst
    MACD = macd #⏸
    MassIndex = massi #🤦🏻‍♂️
    Momentum = mom
    PercentagePriceOscillator = ppo
    RateOfChange = roc
    RelativeStrengthIndex = rsi
    TrueStrengthIndex = tsi
    UltimateOscillator = uo
    WilliamsR = willr

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
