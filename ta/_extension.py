# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

from pandas.core.base import PandasObject


class BasePandasObject(PandasObject):
    """Simple PandasObject Extension
    
    Ensures the DataFrame is not empty and has columns."""
    def __init__(self, data, **kwargs):
        if data.empty:
            return None
        
        total_columns = len(data.columns)
        if total_columns > 0:
            # Lower Case Initial Columns
            data.columns = data.columns.str.lower()
            self._data = data
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
    
    def _valid_df(self, name=None):
        try:
            df = self._data
        except AttributeError as ex:
            msg = f"[X] {ex}: Invalid DataFrame"
            msg = msg + f": {name}" if name else msg
            print(msg)
            # if name:
            #     print(msg + f": {name}")
            # else:
            #     print(msg)
            return None
        return df

    def _valid_pandas(self, panda_object):
        """ Private Method that return True is the panda_object is a DataFrame or Series"""
        return isinstance(panda_object, pd.DataFrame) or isinstance(panda_object, pd.Series)

    ## Overlay Indicators
    def hl2(self, high=None, low=None, **kwargs):
        """Returns the average of two series.

        Args:
            high: None or a Series or DataFrame, optional
                If None, uses local df column: 'high'
            low: None or a Series or DataFrame, optional
                If None, uses local df column: 'low'
            append: bool, kwarg, optional
                If True, appends result to current df
        
        Returns:
            pd.Series: New feature
        """
        df = self._valid_df('hl2')

        # Get the correct columns.
        # Loads current Pandas DataFrame column if None are passed in.
        if self._valid_pandas(high):
            high = high
        else:
            high = df[high] if high in df.columns else df.high
        
        if self._valid_pandas(low):
            low = low
        else:
            low = df[low] if low in df.columns else df.low
        
        # Calculate Result
        hl2 = 0.5 * (high + low)

        # Name & Category
        hl2.name = 'HL2'
        hl2.category = 'overlay'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[hl2.name] = hl2
        
        return hl2


    def hlc3(self, high=None, low=None, close=None, **kwargs):
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
        
        Returns:
            pd.Series: New feature
        """
        df = self._valid_df('hlc3')

        # Get the correct columns.
        # If parameters are pandas, use those and skip df columns
        if self._valid_pandas(high):
            high = high
        else:
            high = df[high] if high in df.columns else df.high
        
        if self._valid_pandas(low):
            low = low
        else:
            low = df[low] if low in df.columns else df.low

        if self._valid_pandas(close):
            close = close
        else:
            close = df[close] if close in df.columns else df.close

        # Calculate Result
        hlc3 = (high + low + close) / 3

        # Name & Category
        hlc3.name = 'HLC3'
        hlc3.category = 'overlay'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[hlc3.name] = hlc3
        
        return hlc3


    def ohlc4(self, open_=None, high=None, low=None, close=None, **kwargs):
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
        
        Returns:
            pd.Series: New feature
        """
        df = self._valid_df('ohlc4')

        # Get the correct columns.
        # If parameters are pandas, use those and skip df columns
        if self._valid_pandas(open_):
            open_ = open_
        else:
            open_ = df[open_] if open_ in df.columns else df.open

        if self._valid_pandas(high):
            high = high
        else:
            high = df[high] if high in df.columns else df.high
        
        if self._valid_pandas(low):
            low = low
        else:
            low = df[low] if low in df.columns else df.low

        if self._valid_pandas(close):
            close = close
        else:
            close = df[close] if close in df.columns else df.close

        # Calculate Result
        ohlc4 = 0.25 * (open_ + high + low + close)

        # Name & Category
        ohlc4.name = 'OHLC4'
        ohlc4.category = 'overlay'

        # If 'append', then add it to the df
        if 'append' in kwargs and kwargs['append']:
            df[ohlc4.name] = ohlc4
        
        return ohlc4


    def decreasing(self, close:str = None, length:int = None, asint:bool = True, **kwargs):
        """Returns if a Series is Decreasing over a certain length.

        Args:
            close(None,pd.Series,pd.DataFrame): optional. If None, uses local df column: 'close'
            length(int): How many 
            asint(bool): True.  Returns zeros and ones.

            append(bool): kwarg, optional.  If True, appends result to current df
        
        Returns:
            pd.Series: New feature
        """
        df = self._valid_df('decreasing')

        # Sanitize Args
        length = length if length and length > 0 else 1

        # Get the correct column
        if self._valid_pandas(close):
            close = close
        else:
            close = df[close] if close in df.columns else df.close

        # Calculate Result
        decreasing = close.diff(length) < 0
        if asint:
            decreasing = decreasing.astype(int)

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


    def increasing(self, close:str = None, length:int = None, asint:bool = True, **kwargs):
        """Returns if a Series is Increasing over a certain length.

        Args:
            close(None,pd.Series,pd.DataFrame): optional.  If None, uses local df column: 'close'
            length(int): How many 
            asint(bool): True.  Returns zeros and ones.

            append(bool): kwarg, optional.  If True, appends result to current df
        
        Returns:
            pd.Series: New feature
        """
        df = self._valid_df('increasing')
        
        # Sanitize Args
        length = length if length and length > 0 else 1

        # Get the correct column
        if self._valid_pandas(close):
            close = close
        else:
            close = df[close] if close in df.columns else df.close

        # Calculate Result
        increasing = close.diff(length) > 0
        if asint:
            increasing = increasing.astype(int)

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

    ## Aliases
    HL2 = hl2
    HLC3 = hlc3
    OHLC4 = ohlc4
    Decreasing = decreasing
    # Increasing = increasing

ta_indicators = list((x for x in dir(pd.DataFrame().ta) if not x.startswith('_') and not x.endswith('_')))
print(f"[i] Loaded {len(ta_indicators)} TA Indicators: {', '.join(ta_indicators)}")