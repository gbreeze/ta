# -*- coding: utf-8 -*-
from ._extension import *

hl2_docs = \
"""
Returns the average of two series.

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

hlc3_docs = \
"""
Returns the average of three series.

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

ohlc4_docs = \
"""
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

median_docs = \
"""
Median Price

Returns the Median of a Series.

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

midpoint_docs = """
Returns the Midpoint of a Series of a certain length.

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

midprice_docs = \
"""
Returns the Midprice of a Series of a certain length.

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

rpn_docs = \
"""
Range Percentage

Returns the Series of values that are a percentage of the absolute difference of two Series.

Args:
    high: None or a Series or DataFrame, optional
        If None, uses local df column: 'high'
    low: None or a Series or DataFrame, optional
        If None, uses local df column: 'low'
    append: bool, kwarg, optional
        If True, appends result to current df

    **kwargs:
        withLow (bool, optional): If true, adds low value to result
        fillna (value, optional): pd.DataFrame.fillna(value)
        fill_method (value, optional): Type of fill method
        append (bool, optional): If True, appends result to current df.

Returns:
    pd.Series: New feature
"""


kurtosis_docs = \
"""
Kurtosis

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

quantile_docs = \
"""
quantile

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

# Momentum Documentation

# Overlap Documentation
AnalysisIndicators.hl2.__doc__ = hl2_docs
AnalysisIndicators.hlc3.__doc__ = hlc3_docs
AnalysisIndicators.ohlc4.__doc__ = ohlc4_docs
AnalysisIndicators.median.__doc__ = median_docs
AnalysisIndicators.midpoint.__doc__ = midpoint_docs
AnalysisIndicators.midprice.__doc__ = midprice_docs
AnalysisIndicators.rpn.__doc__ = rpn_docs

# Performance Documentation


# Statistics Documentation
AnalysisIndicators.kurtosis.__doc__ = kurtosis_docs
AnalysisIndicators.quantile.__doc__ = quantile_docs
