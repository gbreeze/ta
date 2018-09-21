# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

from pandas.core.base import PandasObject
from sys import float_info as sflt

TA_EPSILON = sflt.epsilon

def validate_positive(fn, x, minimum, default=None):
    return fn(x) if x and default and x > minimum else fn(default)


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

    def __call__(self, kind=None, **kwargs):
        try:
            indicator = getattr(self, kind.lower())
        except AttributeError:
            raise ValueError(f"kind='{kind.lower()}' is not valid for {self.__class__.__name__}")
        
        return indicator(**kwargs)


    ## Private Methods
    def _valid_df(self):
        """Validates and returns self._df   ** May be overkill."""
        try:
            df = self._df
        except AttributeError:
            print(f"[X] Not a valid DataFrame.")
            return
        else:
            return df

    # @property
    def defaults(self, value, min_range:int= 0, max_range:int = 100, every:int = 10):
        _levels = [x for x in range(min_range, max_range + 1) if x % every == 0]
        if value:
            for x in _levels:
                self._df[f'{x}'] = x
        else:
            for x in _levels:
                del self._df[f'{x}']



    ## Performance Indicators
    def percent_return(self, close=None, length=None, cummulative=False, percentage=False, offset=None, **kwargs):
        """Returns the Percent Change of a Series.

        Args:
            close (None, pd.Series, optional):
                If None, uses local df column: 'high'
            length (None, int, optional):
                An integer of how periods to compute.  Default is None and one.
            cummulative (bool):
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
        df = self._valid_df()

        if df is not None:
            # Get the correct column.
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close
        else:
            return

        # Validate Arguments
        length = validate_positive(int, length, minimum=1, default=1)
        offset = offset if isinstance(offset, int) else 0
        percent = 100 if percentage else 1

        # Calculate Result
        pct_return = percent * close.pct_change(length)

        if cummulative:
            pct_return = percent * pct_return.cumsum()

        # Offset
        pct_return.shift(offset)

        # Name & Category
        pct_return.name = f"PCTRET_{length}"
        pct_return.category = 'performance'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[pct_return.name] = pct_return
        
        return pct_return


    def log_return(self, close=None, length=None, cummulative=False, percentage=False, offset=None, **kwargs):
        """Returns the Log Return of a Series.

        Args:
            close (None, pd.Series, optional):
                If None, uses local df column: 'high'
            length (None, int, optional):
                An integer of how periods to compute.  Default is None and one.
            cummulative (bool):
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
        df = self._valid_df()

        if df is not None:
            # Get the correct column.
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close
        else:
            return

        # Validate Arguments
        length = validate_positive(int, length, minimum=1, default=1)
        offset = offset if isinstance(offset, int) else 0
        percent = 100 if percentage else 1

        # Calculate Result
        log_return = percent * np.log(close).diff(length)

        if cummulative:
            log_return = log_return.cumsum()

        # Offset
        log_return.shift(offset)

        # Name & Category
        log_return.name = f"LOGRET_{length}"
        log_return.category = 'performance'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[log_return.name] = log_return
        
        return log_return



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
        df = self._valid_df()

        if df is not None:
            # Get the correct column(s).
            if isinstance(high, pd.Series):
                high = high
            else:
                high = df[high] if high in df.columns else df.high
            
            if isinstance(low, pd.Series):
                low = low
            else:
                low = df[low] if low in df.columns else df.low
        else:
            return

        # Validate Arguments
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        hl2 = 0.5 * (high + low)

        # Offset
        hl2.shift(offset)

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
        df = self._valid_df()

        if df is not None:
            # Get the correct column(s).
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
        else:
            return

        # Validate Arguments
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        hlc3 = (high + low + close) / 3

        # Offset
        hlc3.shift(offset)

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
        df = self._valid_df()

        if df is not None:
            # Get the correct column(s).
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
        else:
            return

        # Validate Arguments
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        ohlc4 = 0.25 * (open_ + high + low + close)

        # Offset
        ohlc4.shift(offset)

        # Name & Category
        ohlc4.name = "OHLC4"
        ohlc4.category = 'overlap'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[ohlc4.name] = ohlc4
        
        return ohlc4


    def decreasing(self, close:str = None, length:int = None, asint:bool = True, offset=None, **kwargs):
        """Returns if a Series is Decreasing over a certain length.

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
        df = self._valid_df()

        if df is not None:
            # Get the correct column(s).
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close
        else:
            return

        # Validate Arguments
        length = validate_positive(int, length, minimum=1, default=1)
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        decreasing = close.diff(length) < 0
        if asint:
            decreasing = decreasing.astype(int)
        
        # Offset
        decreasing.shift(offset)

        # Handle fills
        if 'fillna' in kwargs:
            decreasing.fillna(kwargs['fillna'], inplace=True)
        elif 'fill_method' in kwargs:
            decreasing.fillna(method=kwargs['fill_method'], inplace=True)
        
        # Name and Categorize it
        decreasing.name = f"DEC_{length}"
        decreasing.category = 'trend'
        
        # If append, then add it to the df 
        if 'append' in kwargs and kwargs['append']:
            df[decreasing.name] = decreasing

        return decreasing


    def increasing(self, close:str = None, length:int = None, asint:bool = True, offset=None, **kwargs):
        """Returns if a Series is Increasing over a certain length.

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
        df = self._valid_df()

        if df is not None:
            # Get the correct column(s).
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close
        else:
            return

        # Validate arguments
        length = validate_positive(int, length, minimum=1, default=1)
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        increasing = close.diff(length) > 0
        if asint:
            increasing = increasing.astype(int)
        
        # Offset
        increasing.shift(offset)

        # Handle fills
        if 'fillna' in kwargs:
            increasing.fillna(kwargs['fillna'], inplace=True)
        elif 'fill_method' in kwargs:
            increasing.fillna(method=kwargs['fill_method'], inplace=True)
        
        # Name and Categorize it
        increasing.name = f"INC_{length}"
        increasing.category = 'trend'
        
        # If append, then add it to the df 
        if 'append' in kwargs and kwargs['append']:
            df[increasing.name] = increasing

        return increasing


    def midpoint(self, close:str = None, length:int = None, offset=None, **kwargs):
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
        df = self._valid_df()

        if df is not None:
            # Get the correct column(s).
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close
        else:
            return

        # Validate arguments
        length = validate_positive(int, length, minimum=1, default=1)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length
        offset = offset if isinstance(offset, int) else 0

        # Calculate Result
        lowest = close.rolling(length, min_periods=min_periods).min()
        highest = close.rolling(length, min_periods=min_periods).max()
        midpoint = 0.5 * (lowest + highest)
        
        # Offset
        midpoint.shift(offset)

        # Handle fills
        if 'fillna' in kwargs:
            midpoint.fillna(kwargs['fillna'], inplace=True)
        elif 'fill_method' in kwargs:
            midpoint.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        midpoint.name = f"MIDPOINT_{length}"
        midpoint.category = 'overlap'
        
        # If append, then add it to the df 
        if 'append' in kwargs and kwargs['append']:
            df[midpoint.name] = midpoint
            
        return midpoint


    ## Overlap Indicators
    def rpn(self, high=None, low=None, length=None, percentage=None, **kwargs):
        """Returns the Series of values that are a percentage of the absolute difference of two Series.

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
        df = self._valid_df()

        if df is not None:
            # Get the correct column(s).
            if isinstance(high, pd.Series):
                high = high
            else:
                high = df[high] if high in df.columns else df.high
            
            if isinstance(low, pd.Series):
                low = low
            else:
                low = df[low] if low in df.columns else df.low
        else:
            return

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


    def donchian(self, close=None, length:int = None, **kwargs):
        """ donchian """
        df = self._valid_df()

        if df is not None:
            # Get the correct column.
            if isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close
        else:
            return

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
        elif 'fill_method' in kwargs:
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
            
        return dcdf


    def midprice(self, high:str = None, low:str = None, length:int = None, **kwargs):
        """ midprice """
        df = self._valid_df()

        if df is not None:
            # Get the correct column(s).
            if isinstance(low, pd.Series):
                low = low
            else:
                low = df[low] if low in df.columns else df.low

            if isinstance(high, pd.Series):
                high = high
            else:
                high = df[high] if high in df.columns else df.high
        else:
            return

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
        elif 'fill_method' in kwargs:
            midprice.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        midprice.name = f"MIDPRICE_{length}"
        midprice.category = 'overlap'
        
        # If append, then add it to the df 
        if 'append' in kwargs and kwargs['append']:
            df[midprice.name] = midprice
            
        return midprice


    def mom(self, close:str = None, length:int = None, **kwargs):
        """ mom """
        df = self._valid_df()

        if df is not None:
            # Get the correct column.
            if isinstance(close, pd.DataFrame) or isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close
        else:
            return
        
        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=1)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

        # Calculate Result
        mom = close.diff(length)
        
        # Handle fills
        if 'fillna' in kwargs:
            mom.fillna(kwargs['fillna'], inplace=True)
        elif 'fill_method' in kwargs:
            mom.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        mom.name = f"MOM_{length}"
        mom.category = 'momentum'
        
        # If append, then add it to the df 
        if 'append' in kwargs and kwargs['append']:
            df[mom.name] = mom

        return mom



    def roc(self, close:str = None, length:int = None, **kwargs):
        """ roc """
        df = self._valid_df()

        if df is not None:
            # Get the correct column.
            if isinstance(close, pd.DataFrame) or isinstance(close, pd.Series):
                close = close
            else:
                close = df[close] if close in df.columns else df.close
        else:
            return
        
        # Validate arguments
        length = validate_positive(int, length, minimum=0, default=1)
        min_periods = validate_positive(int, kwargs['minperiods']) if 'minperiods' in kwargs else length

        # Calculate Result
        roc = 100 * self.mom(close=close, length=length) / close.shift(length)
        
        # Handle fills
        if 'fillna' in kwargs:
            roc.fillna(kwargs['fillna'], inplace=True)
        elif 'fill_method' in kwargs:
            roc.fillna(method=kwargs['fill_method'], inplace=True)

        # Name and Categorize it
        roc.name = f"ROC_{length}"
        roc.category = 'momentum'
        
        # If append, then add it to the df 
        if 'append' in kwargs and kwargs['append']:
            df[roc.name] = roc

        return roc


    ## Indicator Aliases & Categories
    # Performance
    PctReturn = percent_return
    LogReturn = log_return

    # Overlap
    Decreasing = decreasing
    Increasing = increasing
    HL2 = hl2
    HLC3 = hlc3
    OHLC4 = ohlc4
    Midpoint = midpoint
    Midprice = midprice
    range_percentage = rpn

    # Momentum
    Momentum = mom
    RateOfChange = roc

    # Volume

    # Volatility
    Donchian = donchian

    # Statistics


ta_indicators = list((x for x in dir(pd.DataFrame().ta) if not x.startswith('_') and not x.endswith('_')))
if False:
    print(f"[i] Loaded {len(ta_indicators)} TA Indicators: {', '.join(ta_indicators)}")