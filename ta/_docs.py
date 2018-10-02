# -*- coding: utf-8 -*-
from ._extension import *

# Momentum Documentation
apo_docs = \
""" apo

Not visually the same as TV Chart
"""

macd_docs = \
"""
Moving Average Convergence Divergence

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

massi_docs = \
"""
Mass Index

Not visually the same as TV Chart

"""

mfi_docs = \
"""
Money Flow Index

Incorrect

"""

rsi_docs = \
"""
Relative Strength Index

"""



# Overlap Documentation
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



# Performance Documentation
log_return_docs = \
"""
Log Return with cumulative and offset

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

percent_return_docs = \
"""
Percent Return with Length, Cumulation, Percentage and Offset Attributes

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



# Statistics Documentation
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

mcv_docs = """
Moving Covariance by Rashad (Not visually near it's TradingView Chart)

    Source:
    //Moving Covariance by Rashad. Trading View.
    study(title="Moving Covariance", shorttitle="MCV", overlay=false)
    src = vwap, len = input(30, minval=1, title="Length")
    mean = vwma(src, len)
    stdev = stdev(src, len)
    covariance = (stdev/mean)*100
    plot(covariance, title = "moving covairance", style=line, linewidth = 2, color = red)

Args:
    close (None, pd.Series, optional):
        If None, uses local df column: 'close'
    volume (None, pd.Series, optional):
        If None, uses local df column: 'volume'
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

skew_docs = \
"""
Skew

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

stdev_docs = \
"""
Standard Deviation

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

variance_docs = \
"""
Variance

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



# Trend Documentation
decreasing_docs = \
"""
Decreasing Trend

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

dpo_docs = \
"""
Detrend Price Oscillator (DPO)

Is an indicator designed to remove trend from price and make it easier to
identify cycles.

http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

Args:
    close(None,pd.Series,pd.DataFrame): Optional
        If None, uses local df column: 'close'
    length(int): How many periods to use.
    asint(bool): Default: True.  Returns zeros and ones.

**kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method
    append (bool, optional): If True, appends result to current df.

Returns:
    pd.Series: New feature
"""

increasing_docs = \
"""
Increasing Trend

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



# Volatility Documentation
atr_docs = \
"""
Average True Range

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

bbands_docs = \
"""
Bollinger Bands

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

donchian_docs = \
"""
Donchian Channels

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

kc_docs = \
"""
Keltner Channels

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

true_range_docs = \
"""
True Range

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


# Volume Documentation
ad_docs = \
"""
Accumulation/Distribution

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

cmf_docs = \
"""
Chaikin Money Flow

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

efi_docs = \
"""
Elder's Force Index

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

eom_docs = \
"""
Ease of Movement

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

nvi_docs = \
"""
Negative Volume Index

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

obv_docs = \
"""
On Balance Volume

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

pvol_docs = \
"""
Price Volume

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

pvt_docs = \
"""
Price Volume Trend

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



# Momentum Documentation
AnalysisIndicators.apo.__doc__ = apo_docs
AnalysisIndicators.macd.__doc__ = macd_docs
AnalysisIndicators.massi.__doc__ = massi_docs
AnalysisIndicators.mfi.__doc__ = mfi_docs
AnalysisIndicators.rsi.__doc__ = rsi_docs

# Overlap Documentation
AnalysisIndicators.hl2.__doc__ = hl2_docs
AnalysisIndicators.hlc3.__doc__ = hlc3_docs
AnalysisIndicators.ohlc4.__doc__ = ohlc4_docs
AnalysisIndicators.median.__doc__ = median_docs
AnalysisIndicators.midpoint.__doc__ = midpoint_docs
AnalysisIndicators.midprice.__doc__ = midprice_docs
AnalysisIndicators.rpn.__doc__ = rpn_docs

# Performance Documentation
AnalysisIndicators.log_return.__doc__ = log_return_docs
AnalysisIndicators.percent_return.__doc__ = percent_return_docs

# Statistics Documentation
AnalysisIndicators.kurtosis.__doc__ = kurtosis_docs
AnalysisIndicators.mcv.__doc__ = mcv_docs
AnalysisIndicators.quantile.__doc__ = quantile_docs
AnalysisIndicators.skew.__doc__ = skew_docs
AnalysisIndicators.stdev.__doc__ = stdev_docs
AnalysisIndicators.variance.__doc__ = variance_docs

# Trend Documentation
AnalysisIndicators.decreasing.__doc__ = decreasing_docs
AnalysisIndicators.dpo.__doc__ = dpo_docs
AnalysisIndicators.increasing.__doc__ = increasing_docs

# Volatility Documentation
AnalysisIndicators.atr.__doc__ = atr_docs
AnalysisIndicators.bbands.__doc__ = bbands_docs
AnalysisIndicators.donchian.__doc__ = donchian_docs
AnalysisIndicators.kc.__doc__ = kc_docs
AnalysisIndicators.true_range.__doc__ = true_range_docs

# Volume Documentation
AnalysisIndicators.ad.__doc__ = ad_docs
AnalysisIndicators.cmf.__doc__ = cmf_docs
AnalysisIndicators.efi.__doc__ = efi_docs
AnalysisIndicators.eom.__doc__ = eom_docs
AnalysisIndicators.nvi.__doc__ = nvi_docs
AnalysisIndicators.obv.__doc__ = obv_docs
AnalysisIndicators.pvol.__doc__ = pvol_docs
AnalysisIndicators.pvt.__doc__ = pvt_docs