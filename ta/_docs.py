# -*- coding: utf-8 -*-
from ._extension import *

ai_ohlc4 = """
Calculates and returns the average of four series.

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

AnalysisIndicators.ohlc4.__doc__ = ai_ohlc4