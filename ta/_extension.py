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
            self._data = data
        else:
            raise AttributeError(f"[X] No columns!")
    
    def __call__(self, kind, *args, **kwargs):
        raise NotImplementedError()


@pd.api.extensions.register_dataframe_accessor('ta')
class AnalysisIndicators(BasePandasObject):
    """AnalysisIndicators registers the extension 'ta' to the DataFrame.
    
    All Indicators can be called one of two ways:
    Given a TimeSeries DataFrame called df with lower case column names. ie. open, high, lose, close, volume

    Example:
    df = pd.read_csv('AAPL.csv', index_col='date', parse_dates=True, dtype=float, infer_datetime_format=False, keep_date_col=True)

    Calling HL2:
    * hl2 = df.ta.hl2()
    * hl2 = df.ta.HL2()
    * hl2 = df.ta(kind='hl2')

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
            if name:
                print(msg + f": {name}")
            else:
                print(msg)
            return None
        return df


    ## Indicators
    def hl2(self, high=None, low=None, **kwargs):
        """ hl2 = (high + low) / 2 """
        df = self._valid_df('hl2')

        # Get the correct columns
        if isinstance(high, pd.DataFrame) or isinstance(high, pd.Series):
            high = high
        else:
            high = df[high] if high in df.columns else df.high
        
        if isinstance(low, pd.DataFrame) or isinstance(low, pd.Series):
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



    ## Aliases
    HL2 = hl2