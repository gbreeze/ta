# -*- coding: utf-8 -*-
import time
import math
import numpy as np
import pandas as pd

from pandas.core.base import PandasObject
from sys import float_info as sflt

TA_EPSILON = sflt.epsilon

def validate_positive(fn, x, minimum, default):
    return fn(x) if x and default and x > minimum and default > minimum else fn(default)


def signed_series(series:pd.Series, initial:int = None):
    """Returns a Signed Series with or without an initial value"""
    sign = series.diff(1)
    sign[sign > 0] = 1
    sign[sign < 0] = -1
    sign.iloc[0] = initial
    return sign


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
    # length = validate_positive(int, length, minimum=0, default=1)
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

        # Run the indicator
        if timed:
            stime = time.time()
        indicator = fn(**kwargs)
        if timed:
            time_diff = time.time() - stime
            ms = time_diff * 1000
            indicator.timed = f"{ms:2.3f} ms ({time_diff:2.3f} s)"
        if alias:
            indicator.alias = f"{alias}"
        return indicator


    # @property
    def defaults(self, value, min_range:int= 0, max_range:int = 100, every:int = 10):
        _levels = [x for x in range(min_range, max_range + 1) if x % every == 0]
        if value:
            for x in _levels:
                self._df[f'{x}'] = x
        else:
            for x in _levels:
                del self._df[f'{x}']



    ## Momentum Indicators
    def apo(self, close=None, fast:int = None, slow:int = None, **kwargs):
        """ apo """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        fast = validate_positive(int, fast, minimum=0, default=12)
        slow = validate_positive(int, slow, minimum=0, default=26)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else fast

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

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[apo.name] = apo

        return apo


    def bop(self, open_:str = None, high:str = None, low:str = None, close:str = None, percentage:bool = False, **kwargs):
        """ bop """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(open_, pd.Series):
                open_ = open_
            else:
                open_ = df[open_] if open_ in df.columns else df.open

            if isinstance(high, pd.Series):
                high_ = high
            else:
                high_ = df[high] if high in df.columns else df.high

            if isinstance(low, pd.Series):
                low_ = low
            else:
                low_ = df[low] if low in df.columns else df.low

            if isinstance(close, pd.Series):
                close_ = close
            else:
                close_ = df[close] if close in df.columns else df.close

        # Validate arguments
        # length = validate_positive(int, length, minimum=0, default=1)
        percent = 100 if percentage else 1

        # Calculate Result
        close_open_range = close_ - open_
        high_log_range = high_ - low_
        bop = percent * close_open_range / high_log_range

        # Handle fills
        if 'fillna' in kwargs:
            bop.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            bop.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        bop.name = f"BOP"
        bop.category = 'momentum'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[bop.name] = bop

        return bop


    def cci(self, high:str = None, low:str = None, close:str = None, length:int = None, c:float = None, **kwargs):
        """ cci """
        df = self._df

        # Get the correct column.
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
        length = validate_positive(int, length, minimum=0, default=20)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length
        c = validate_positive(float, c, minimum=0, default=0.015)

        # Calculate Result
        def mad(series):
            """Mean Absolute Deviation"""
            return np.fabs(series - series.mean()).mean()

        typical_price = self.hlc3(high=high, low=low, close=close)
        mean_typical_price = typical_price.rolling(length, min_periods=min_periods).mean()
        mad_typical_price = typical_price.rolling(length).apply(mad, raw=True)

        cci = (typical_price - mean_typical_price) / (c * mad_typical_price)

        # Handle fills
        if 'fillna' in kwargs:
            cci.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            cci.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        # bop.name = f"BOP_{length}"
        cci.name = f"CCI_{length}_{c}"
        cci.category = 'momentum'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[cci.name] = cci

        return cci


    def macd(self, close=None, fast:int = None, slow:int = None, signal:int = None, **kwargs):
        """Moving Average Convergence Divergence

        Returns a DataFrame with high, mid, and low values.  The high channel is max()
        and the low channel is the min() over a rolling period length of the source.
        The mid is the average of the high and low channels.

        Args:
            close(None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            length(int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        fast = validate_positive(int, fast, minimum=0, default=12)
        fast_min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else fast

        slow = validate_positive(int, slow, minimum=0, default=26)
        slow_min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else slow

        signal = validate_positive(int, signal, minimum=0, default=9)
        signal_min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else signal

        # Calculate Result
        fastma = close.ewm(span=fast, min_periods=fast_min_periods).mean()
        slowma = close.ewm(span=slow, min_periods=slow_min_periods).mean()
        macd = fastma - slowma

        signalma = macd.ewm(span=signal, min_periods=signal_min_periods).mean()
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


    def massi(self, high:str = None, low:str = None, single=None, double=None, **kwargs):
        """Mass Index
        
        Incorrect

        """
        df = self._df

        # Get the correct column.
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

        # Validate arguments
        single = validate_positive(int, single, minimum=0, default=9)
        double = validate_positive(int, double, minimum=0, default=25)

        # Calculate Result
        hl_range = high - low
        hl_ema1 = hl_range.ewm(span=single, min_periods=single).mean()
        hl_ema2 =  hl_ema1.ewm(span=single, min_periods=single).mean()

        mass = hl_ema1 / hl_ema2
        massi = mass.rolling(double, min_periods=double).sum()

        # Handle fills
        if 'fillna' in kwargs:
            massi.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            massi.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        # bop.name = f"BOP_{length}"
        massi.name = f"MASSI_{single}_{double}"
        massi.category = 'momentum'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[massi.name] = massi

        return massi


    def mfi(self, high:str = None, low:str = None, close:str = None, volume:str = None, length:int = None, drift:int = None, **kwargs):
        """Money Flow Index

        Incorrect
        
        """
        df = self._df

        # Get the correct column.
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
        length = validate_positive(int, length, minimum=0, default=14)
        drift = validate_positive(int, drift, minimum=0, default=1)

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


    def mom(self, close:str = None, length:int = None, **kwargs):
        """ mom """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=1)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

        # Calculate Result
        mom = close.diff(length)

        # Handle fills
        if 'fillna' in kwargs:
            mom.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            mom.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        mom.name = f"MOM_{length}"
        mom.category = 'momentum'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[mom.name] = mom

        return mom


    def ppo(self, close:str = None, fast:int = None, slow:int = None, **kwargs):
        """ ppo """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.DataFrame) or isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        fast = validate_positive(int, fast, minimum=0, default=12)
        slow = validate_positive(int, slow, minimum=0, default=26)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else fast

        # Calculate Result
        fastma = close.rolling(fast, min_periods=min_periods).mean()
        slowma = close.rolling(slow, min_periods=min_periods).mean()
        ppo = 100 * (fastma - slowma) / slowma

        # Handle fills
        if 'fillna' in kwargs:
            ppo.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            ppo.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        ppo.name = f"PPO_{fast}_{slow}"
        ppo.category = 'momentum'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[ppo.name] = ppo

        return ppo


    def roc(self, close:str = None, length:int = None, **kwargs):
        """ roc """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.DataFrame) or isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=1)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

        # Calculate Result
        roc = 100 * self.mom(close=close, length=length) / close.shift(length)

        # Handle fills
        if 'fillna' in kwargs:
            roc.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            roc.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        roc.name = f"ROC_{length}"
        roc.category = 'momentum'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[roc.name] = roc

        return roc


    def rsi(self, close:str = None, length:int = None, drift:int = None, **kwargs):
        """Relative Strength Index
        
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.DataFrame) or isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=14)
        drift = validate_positive(int, drift, minimum=0, default=1)

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


    def willr(self, high:str = None, low:str = None, close:str = None, length:int = None, **kwargs):
        """ willr """
        df = self._df

        # Get the correct column.
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
        length = validate_positive(int, length, minimum=0, default=14)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

        # Calculate Result
        lowest_low = low.rolling(length, min_periods=min_periods).min()
        highest_high = high.rolling(length, min_periods=min_periods).max()

        willr = 100 * ((close - lowest_low) / (highest_high - lowest_low) - 1)

        # Handle fills
        if 'fillna' in kwargs:
            willr.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            willr.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        willr.name = f"WILLR_{length}"
        willr.category = 'momentum'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[willr.name] = willr

        return willr


    ## Overlap Indicators
    def hl2(self, high=None, low=None, offset=None, **kwargs):
        """Returns the average of two series.

        Args:
            high: None or a Series or DataFrame, optional
                If None, uses local df column: 'high'
            low: None or a Series or DataFrame, optional
                If None, uses local df column: 'low'
            append: bool, kwarg, optional
                If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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

        # Validate Arguments
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        hl2 = 0.5 * (high + low)

        # Offset
        hl2 = hl2.shift(offset)

        # Name & Category
        hl2.name = "HL2"
        hl2.category = 'overlap'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[hl2.name] = hl2

        return hl2


    def hlc3(self, high=None, low=None, close=None, offset=None, **kwargs):
        """Returns the average of three series.

        Args:
            high: None or a Series or DataFrame, optional
                If None, uses local df column: 'high'
            low: None or a Series or DataFrame, optional
                If None, uses local df column: 'low'
            close: None or a Series or DataFrame, optional
                If None, uses local df column: 'close'
            append: bool, kwarg, optional
                If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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

        # Validate Arguments
        offset = offset if isinstance(offset, int) else 0

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


    def ohlc4(self, open_=None, high=None, low=None, close=None, offset=None, **kwargs):
        """Calculates and returns the average of four series.

        Args:
            open_: None or a Series or DataFrame, optional
                If None, uses local df column: 'open'
            high: None or a Series or DataFrame, optional
                If None, uses local df column: 'high'
            low: None or a Series or DataFrame, optional
                If None, uses local df column: 'low'
            close: None or a Series or DataFrame, optional
                If None, uses local df column: 'close'
            append: bool, kwarg, optional
                If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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

        # Validate Arguments
        offset = offset if isinstance(offset, int) else 0

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


    def median(self, close=None, length=None, cumulative:bool = False, offset:int = None, **kwargs):
        """Median Price

        Returns the Log Return of a Series.

        Args:
            close (None, pd.Series, optional):
                If None, uses local df column: 'high'
            length (None, int, optional):
                An integer of how periods to compute.  Default is None and one.
            cumulative (bool):
                Default: False.  If True, returns the cummulative returns
            offset (None, int, optional):
                An integer on how to shift the Series.  Default is None and zero.

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate Arguments
        length = validate_positive(int, length, minimum=1, default=5)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length
        offset = offset if isinstance(offset, int) else 0

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


    # def midpoint(self, close:str = None, length:int = None, offset=None, **kwargs):
    def midpoint(self, close:str = None, length:int = None, **kwargs):
        """Returns the Midpoint of a Series of a certain length.

        Args:
            close (None, str, pd.Series, optional):
                pd.Series: A seperate Series not in the current DataFrame.
                str: Looksup column in DataFrame under 'str' name.
                None: Default.  Uses current DataFrame column 'close'.
            length (int): Lookback length. Defaults to 1.

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=1)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length
        offset = offset if isinstance(offset, int) else 0

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


    def midprice(self, high:str = None, low:str = None, length:int = None, **kwargs):
        """ midprice """
        df = self._df

        # Get the correct column(s).
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

        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=1)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

        # Calculate Result
        lowest_low = low.rolling(length, min_periods=min_periods).min()
        highest_high = high.rolling(length, min_periods=min_periods).max()
        midprice = 0.5 * (lowest_low + highest_high)

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


    def rpn(self, high=None, low=None, length=None, percentage=None, **kwargs):
        """Range Percentage

        Returns the Series of values that are a percentage of the absolute difference of two Series.

        Args:
            high: None or a Series or DataFrame, optional
                If None, uses local df column: 'high'
            low: None or a Series or DataFrame, optional
                If None, uses local df column: 'low'
            append: bool, kwarg, optional
                If True, appends result to current df

            **kwargs:
                addLow (bool, optional): If true, adds low value to result
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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


        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=1)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length
        percentage = validate_positive(float, percentage, minimum=0.0, default=0.1)

        # Calculate Result
        highest_high = high.rolling(length, min_periods=min_periods).max()
        lowest_low = low.rolling(length, min_periods=min_periods).min()
        abs_range = (highest_high - lowest_low).abs()

        rp = percentage * abs_range
        if 'addLow' in kwargs and kwargs['addLow']:
            rp += low

        # Name & Category
        rp.name = f"RP_{length}_{percentage}"
        rp.category = 'overlap'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[rp.name] = rp

        return rp


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
        """Log Return with cumulative and offset

        Returns the Log Return of a Series.

        Args:
            close (None, pd.Series, optional):
                If None, uses local df column: 'high'
            length (None, int, optional):
                An integer of how periods to compute.  Default is None and one.
            cumulative (bool):
                Default: False.  If True, returns the cummulative returns
            offset (None, int, optional):
                An integer on how to shift the Series.  Default is None and zero.

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate Arguments
        length = validate_positive(int, length, minimum=0, default=1)
        offset = offset if isinstance(offset, int) else 0
        percent = 100 if percent else 1

        # Calculate Result
        log_return = percent * np.log(close).diff(periods=length)

        if cumulative:
            log_return = log_return.cumsum()

        # Offset
        log_return.shift(offset)

        # Name & Category
        log_return.name = f"{'CUM_' if cumulative else ''}LOGRET_{length}"
        log_return.category = 'performance'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[log_return.name] = log_return

        return log_return


    def percent_return(self, close=None, length=None, cumulative:bool = False, percent:bool = False, offset:int = None, **kwargs):
        """Percent Return with Length, Cumulation, Percentage and Offset Attributes

        Returns the Percent Change of a Series.

        Args:
            close (None, pd.Series, optional):
                If None, uses local df column: 'high'
            length (None, int, optional):
                An integer of how periods to compute.  Default is None and one.
            cumulative (bool):
                Default: False.  If True, returns the cummulative returns
            offset (None, int, optional):
                An integer on how to shift the Series.  Default is None and zero.

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate Arguments
        length = validate_positive(int, length, minimum=0, default=1)
        offset = offset if isinstance(offset, int) else 0
        percent = 100 if percent else 1

        # Calculate Result
        pct_return = percent * close.pct_change(length)

        if cumulative:
            pct_return = percent * pct_return.cumsum()

        # Offset
        pct_return = pct_return.shift(offset)

        # Name & Category
        pct_return.name = f"{'CUM_' if cumulative else ''}PCTRET_{length}"
        pct_return.category = 'performance'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[pct_return.name] = pct_return

        return pct_return


    ## Statistics Indicators
    def kurtosis(self, close=None, length=None, **kwargs):
        """Kurtosis

        Returns the Kurtosis of a Series.

        Args:
            close (None, pd.Series, optional):
                If None, uses local df column: 'high'
            length (None, int, optional):
                An integer of how periods to compute.  Default is None and one.

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate Arguments
        length = validate_positive(int, length, minimum=3, default=30)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

        # Calculate Result
        kurtosis = close.rolling(length, min_periods=min_periods).kurt()

        # Name & Category
        kurtosis.name = f"KURT_{length}"
        kurtosis.category = 'statistics'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[kurtosis.name] = kurtosis

        return kurtosis


    def quantile(self, close=None, length=None, q:float = None, **kwargs):
        """quantile

        Returns the quantile of a Series.

        Args:
            close (None, pd.Series, optional):
                If None, uses local df column: 'high'
            length (None, int, optional):
                An integer of how periods to compute.  Default is None and one.

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate Arguments
        length = validate_positive(int, length, minimum=3, default=30)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length
        q = float(q) if q and q > 0 and q <= 1 else 0.5
        # q = validate_positive(float, q, minimum=0, default=0.5) and float(q) <= 1.0

        # Calculate Result
        quantile = close.rolling(length, min_periods=min_periods).quantile(q)

        # Name & Category
        quantile.name = f"QTL_{length}_{q}"
        quantile.category = 'statistics'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[quantile.name] = quantile

        return quantile


    def skew(self, close=None, length=None, **kwargs):
        """Skew

        Returns the Skew of a Series.

        Args:
            close (None, pd.Series, optional):
                If None, uses local df column: 'high'
            length (None, int, optional):
                An integer of how periods to compute.  Default is None and one.

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate Arguments
        length = validate_positive(int, length, minimum=3, default=30)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

        # Calculate Result
        skew = close.rolling(length, min_periods=min_periods).skew()

        # Name & Category
        skew.name = f"SKEW_{length}"
        skew.category = 'statistics'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[skew.name] = skew

        return skew


    def stdev(self, close=None, length=None, **kwargs):
        """Standard Deviation

        Returns the Standard Deviations of a Series.

        Args:
            close (None, pd.Series, optional):
                If None, uses local df column: 'high'
            length (None, int, optional):
                An integer of how periods to compute.  Default is None and one.

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate Arguments
        length = validate_positive(int, length, minimum=2, default=30)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

        # Calculate Result
        stdev = self.variance(length=length).apply(np.sqrt)

        # Name & Category
        stdev.name = f"STDEV_{length}"
        stdev.category = 'statistics'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[stdev.name] = stdev

        return stdev


    def variance(self, close=None, length=None, **kwargs):
        """Variance

        Returns the Variances of a Series.

        Args:
            close (None, pd.Series, optional):
                If None, uses local df column: 'high'
            length (None, int, optional):
                An integer of how periods to compute.  Default is None and one.
            cumulative (bool):
                Default: False.  If True, returns the cummulative returns
            offset (None, int, optional):
                An integer on how to shift the Series.  Default is None and zero.

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate Arguments
        length = validate_positive(int, length, minimum=2, default=30)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

        # Calculate Result
        variance = close.rolling(length, min_periods=min_periods).var()

        # Name & Category
        variance.name = f"VAR_{length}"
        variance.category = 'statistics'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[variance.name] = variance

        return variance



    ## Trend Indicators
    def decreasing(self, close:str = None, length:int = None, asint:bool = True, offset=None, **kwargs):
        """Decreasing Trend

        Returns if a Series is Decreasing over a certain length.

        Args:
            close(None,pd.Series,pd.DataFrame): optional. If None, uses local df column: 'close'
            length(int): How many periods long.
            asint(bool): True.  Returns zeros and ones.

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate Arguments
        length = validate_positive(int, length, minimum=1, default=1)
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        decreasing = close.diff(length) < 0
        if asint:
            decreasing = decreasing.astype(int)

        # Offset
        decreasing = decreasing.shift(offset)

        # Handle fills
        if 'fillna' in kwargs:
            decreasing.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            decreasing.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        decreasing.name = f"DEC_{length}"
        decreasing.category = 'trend'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[decreasing.name] = decreasing

        return decreasing


    def increasing(self, close:str = None, length:int = None, asint:bool = True, offset=None, **kwargs):
        """Increasing Trend

        Returns if a Series is Increasing over a certain length.

        Args:
            close(None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            length(int): How many
            asint(bool): True.  Returns zeros and ones.

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        length = validate_positive(int, length, minimum=1, default=1)
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        increasing = close.diff(length) > 0
        if asint:
            increasing = increasing.astype(int)

        # Offset
        increasing = increasing.shift(offset)

        # Handle fills
        if 'fillna' in kwargs:
            increasing.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            increasing.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        increasing.name = f"INC_{length}"
        increasing.category = 'trend'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[increasing.name] = increasing

        return increasing


    ## Volatility Indicators
    def atr(self, high=None, low=None, close=None, length=None, mamode:str = None, **kwargs):
        """Average True Range

        Returns a Series of the Average True Range.

        Args:
            close (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            volume (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'volume'
            signed (bool): True.  Returns zeros and ones.
            offset (int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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
        length = validate_positive(int, length, minimum=0, default=14)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length
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
        """Bollinger Bands

        Returns a DataFrame with high, mid, and low values.  The high channel is max()
        and the low channel is the min() over a rolling period length of the source.
        The mid is the average of the high and low channels.

        Args:
            close(None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            length(int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=20)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length
        stdev = validate_positive(int, stdev, minimum=0, default=2)

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
        """Donchian Channels

        Returns a DataFrame with high, mid, and low values.  The high channel is max()
        and the low channel is the min() over a rolling period length of the source.
        The mid is the average of the high and low channels.

        Args:
            close(None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            length(int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column.
        if df is None: return
        else:
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close

        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=20)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

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
        """Keltner Channels

        Returns a DataFrame with high, mid, and low values.  The high channel is max()
        and the low channel is the min() over a rolling period length of the source.
        The mid is the average of the high and low channels.

        Args:
            close(None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            length(int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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
        length = validate_positive(int, length, minimum=0, default=20)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length
        scalar = validate_positive(float, scalar, minimum=0, default=2)
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
        df = self._df
        
        # if df is None: return
        # else:
        #     # Get the correct column.
        #     if isinstance(high, pd.Series):
        #         high = high
        #     else:
        #         high = df[high] if high in df.columns else df.high

        #     if isinstance(low, pd.Series):
        #         low = low
        #     else:
        #         low = df[low] if low in df.columns else df.low

        #     if isinstance(close, pd.Series):
        #         close = close
        #     else:
        #         close = df[close] if close in df.columns else df.close
        # print(f"df:\n{df.head()}")
        # print(f"{high.head()}\n{low.head()}\n{close.head()}")
        return _stoch(df, high=high, low=low, close=close, fast_k=fast_k, slow_k=slow_k, slow_d=slow_d, **kwargs)

    def true_range(self, high=None, low=None, close=None, length=None, **kwargs):
        """True Range

        Returns a Series of the product of Price and Volume.

        Args:
            close (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            volume (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'volume'
            signed (bool): True.  Returns zeros and ones.
            offset (int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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
        length = validate_positive(int, length, minimum=0, default=1)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

        # Calculate Result
        prev_close = close.shift(1)
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
        """Accumulation/Distribution

        Returns a Series of the product of Price and Volume.

        Args:
            high (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'high'
            low (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'low'
            close (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            volume (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'volume'
            open_ (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'open_'
            signed (bool): True.  Returns zeros and ones.
            offset (int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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

                ad - close - open_  # AD with Open
            else:                
                ad = 2 * close - high - low  # AD with High, Low, Close

        # Validate arguments
        offset = offset if isinstance(offset, int) else 0
        # min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else offset

        # Calculate Result
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

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[ad.name] = ad

        return ad


    def efi(self, close=None, volume=None, length=None, mamode:str = None, offset:int = None, drift:int = None, **kwargs):
        """Elder's Force Index

        Returns a Series of the product of Price and Volume.

        Args:
            close (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            volume (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'volume'
            signed (bool): True.  Returns zeros and ones.
            offset (int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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

        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=1)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length
        drift = validate_positive(int, drift, minimum=0, default=1)
        offset = offset if isinstance(offset, int) else 0

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

        # If append, then add it to the df
        # if 'append' in kwargs and kwargs['append']:
        if kwargs.pop('append', False):
            df[efi.name] = efi

        return efi


    def cmf(self, high=None, low=None, close=None, volume=None, length=None, open_=None, offset:int = None, **kwargs):
        """Chaikin Money Flow

        Returns a Series of the product of Price and Volume.

        Args:
            close (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            volume (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'volume'
            signed (bool): True.  Returns zeros and ones.
            offset (int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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
                    open_ = df[open_] if open_open_ in df.columns else df.open
                
                ad = close - open_  # AD with Open
            else:
                ad = 2 * close - high - low  # AD with High, Low, Close

        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=1)
        min_periods = int(kwargs['minperiods']) if 'minperiods' in kwargs else length
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
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

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[cmf.name] = cmf

        return cmf


    def eom(self, high=None, low=None, close=None, volume=None, length=None, divisor:int = None, ease:int = None, offset:int = None, **kwargs):
        """Ease of Movement

        Returns a Series of the product of Price and Volume.

        Args:
            close (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            volume (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'volume'
            signed (bool): True.  Returns zeros and ones.
            offset (int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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
        length = validate_positive(int, length, minimum=0, default=1)
        min_periods = int(kwargs['minperiods']) if 'minperiods' in kwargs else length
        divisor = divisor if divisor and divisor > 0 else 100000000
        ease = int(ease) if ease and ease > 0 else 1
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        hl_range = high - low
        distance = self.hl2(high=high, low=low) - self.hl2(high=high.shift(ease), low=low.shift(ease))
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

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[eom.name] = eom

        return eom


    def nvi(self, close=None, volume=None, length:int = None, initial:int = None, signed:bool = True, offset:int = None, **kwargs):
        """Negative Volume Index

        Returns a Series of the product of Price and Volume.

        Args:
            close (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            volume (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'volume'
            signed (bool): True.  Returns zeros and ones.
            offset (int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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

        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=1)
        initial = initial if initial and initial > 0 else 1000
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        roc = self.roc(close=close)
        signed_volume = signed_series(volume, initial=1)
        nvi = signed_volume[signed_volume < 0].abs() * roc
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

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[nvi.name] = nvi

        return nvi


    def obv(self, close=None, volume=None, offset:int = None, **kwargs):
        """On Balance Volume

        Returns a Series of the product of Price and Volume.

        Args:
            close (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            volume (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'volume'
            signed (bool): True.  Returns zeros and ones.
            offset (int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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

        # Validate arguments
        offset = offset if isinstance(offset, int) else 0

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

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[obv.name] = obv

        return obv


    def pvol(self, close:str = None, volume:str = None, signed:bool = True, offset:int = None, **kwargs):
        """Price Volume

        Returns a Series of the product of Price and Volume.

        Args:
            close (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            volume (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'volume'
            signed (bool): True.  Returns zeros and ones.
            offset (int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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

        # Validate arguments
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        if signed:
            pvol = signed_series(close, 1) * close * volume
        else:
            pvol = close * volume

        # Offset
        pvol = pvol.shift(offset)

        # Handle fills
        if 'fillna' in kwargs:
            pvol.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            pvol.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        pvol.name = f"PVOL"
        pvol.category = 'volume'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[pvol.name] = pvol

        return pvol


    def pv_trend(self, close=None, volume=None, length=None, offset:int = None, **kwargs):
        """Price Volume Trend

        Returns a Series of the product of Price and Volume.

        Args:
            close (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            volume (None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'volume'
            signed (bool): True.  Returns zeros and ones.
            offset (int): How many

            append(bool): kwarg, optional.  If True, appends result to current df

            **kwargs:
                fillna (value, optional): pd.DataFrame.fillna(value)
                fill_method (value, optional): Type of fill method
                append (bool, optional): If True, appends result to current df.

        Returns:
            pd.Series: New feature
        """
        df = self._df

        # Get the correct column(s).
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

        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=1)
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        pv = self.roc(close=close, length=length) * volume
        pvt = pv.cumsum()

        # Offset
        pvt = pvt.shift(offset)

        # Handle fills
        if 'fillna' in kwargs:
            pvt.fillna(kwargs['fillna'], inplace=True)
        if 'fill_method' in kwargs:
            pvt.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        pvt.name = f"PVT_{length}"
        pvt.category = 'volume'

        # If append, then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[pvt.name] = pvt

        return pvt


    ## Indicator Aliases & Categories
    # Momentum
    AbsolutePriceOscillator = apo
    BalanceOfPower = bop
    CommodityChannelIndex = cci
    MACD = macd
    MassIndex = massi
    Momentum = mom
    PercentagePriceOscillator = ppo
    RateOfChange = roc
    RelativeStrengthIndex = rsi
    WilliamsR = willr

    # Overlap
    HL2 = hl2
    HLC3 = hlc3
    OHLC4 = ohlc4
    Median = median
    Midpoint = midpoint
    Midprice = midprice
    RangePercentage = rpn

    # Performance
    LogReturn = log_return
    PctReturn = percent_return

    # Statistics
    Kurtosis = kurtosis
    Quantile = quantile
    Skew = skew
    StandardDeviation = stdev
    Variance = variance

    # Trend
    Decreasing = decreasing
    Increasing = increasing

    # Volatility
    AverageTrueRange = atr
    BollingerBands = bbands
    DonchianChannels = donchian
    KeltnerChannels = kc
    TrueRange = true_range

    # Volume
    AccumDist = ad
    ChaikinMoneyFlow = cmf
    EldersForceIndex = efi
    EaseOfMovement = eom
    NegativeVolumeIndex = nvi
    OnBalanceVolume = obv
    PriceVolume = pvol
    PriceVolumeTrend = pv_trend


ta_indicators = list((x for x in dir(pd.DataFrame().ta) if not x.startswith('_') and not x.endswith('_')))
if False:
    print(f"[i] Loaded {len(ta_indicators)} TA Indicators: {', '.join(ta_indicators)}")