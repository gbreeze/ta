# -*- coding: utf-8 -*-
# from ._extension import *
from .momentum import *
from .overlap import *
from .performance import *
from .statistics import *
from .trend import *
from .volatility import *
from .volume import *



# Momentum Documentation
ao_docs = \
"""Awesome Oscillator (AO)

The Awesome Oscillator is an indicator used to measure a security's momentum. 
AO is generally used to affirm trends or to anticipate possible reversals.

Calculation:
    Default Inputs:
        fast: 5, slow: 34
    SMA = Simple Moving Average
    median = (high + low) / 2
    AO = SMA(median, fast) - SMA(median, slow)

Sources:
    https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)
    https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    fast (int): The short period.  Default: 5
    slow (int): The long period.   Default: 34
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


apo_docs = \
"""Absolute Price Oscillator (APO)

The Absolute Price Oscillator is an indicator used to measure a security's
momentum.  It is simply the difference of two Exponential Moving Averages
(EMA) of two different periods.  Note: APO and MACD lines are equivalent.

Calculation:
    Default Inputs:
        fast: 12, slow: 26
    EMA = Exponential Moving Average
    APO = EMA(close, fast) - EMA(close, slow)

Args:
    close (pd.Series): Series of 'close's
    fast (int): The short period.  Default: 12
    slow (int): The long period.   Default: 26
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


bop_docs = \
"""Balance of Power (BOP)

Balance of Power measure the market strength of buyers against sellers.

Calculation:
    BOP = (close - open) / (high - low)

Sources:
    http://www.worden.com/TeleChartHelp/Content/Indicators/Balance_of_Power.htm

Args:
    open (pd.Series): Series of 'open's
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


cci_docs = \
"""Commodity Channel Index (CCI)

Commodity Channel Index is a momentum oscillator used to primarily identify
overbought and oversold levels relative to a mean.

Sources:
    https://www.tradingview.com/wiki/Commodity_Channel_Index_(CCI)

Calculation:
    SMA = Simple Moving Average
    MAD = Mean Absolute Deviation
    tp = typical_price = hlc3 = (high + low + close) / 3
    mean_tp = SMA(tp, length)
    mad_tp = MAD(tp, length)
    CCI = (tp - mean_tp) / (c * mad_tp)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 20
    c (float):  Scaling Constant.  Default: 0.015
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


macd_docs = \
"""Moving Average Convergence Divergence (MACD)

The MACD is a popular indicator to that is used to identify a security's trend.
While APO and MACD are the same calculation, MACD also returns two more series
called Signal and Histogram.  The Signal is an EMA of MACD and the Histogram is
the difference of MACD and Signal.

Calculation:
    Default Inputs:
        fast: 12, slow: 26, signal: 9
    EMA = Exponential Moving Average
    MACD = EMA(close, fast) - EMA(close, slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal

Sources:
    https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)

Args:
    close(pandas.Series): Series of 'close's
    fast(int): The short period.  Default: 12
    slow(int): The long period.   Default: 26
    signal(int): The signal period.   Default: 9
    offset(int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: macd, histogram, signal columns
"""


mfi_docs = \
"""Money Flow Index (MFI)

Money Flow Index is an oscillator indicator that is used to measure buying and
selling pressure by utilizing both price and volume.

Sources:
    https://www.tradingview.com/wiki/Money_Flow_(MFI)

Calculation:
    tp = typical_price = hlc3 = (high + low + close) / 3
    rmf = raw_money_flow = tp * volume

    pmf = pos_money_flow = SUM(rmf, length) if tp.diff(drift) > 0 else 0
    nmf = neg_money_flow = SUM(rmf, length) if tp.diff(drift) < 0 else 0

    MFR = money_flow_ratio = pmf / nmf
    MFI = money_flow_index = 100 * pmf / (pmf + nmf)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume'
    length (int): The sum period.  Default: 14
    drift (int): The difference period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


mom_docs = \
"""Momentum (MOM)

Momentum is an indicator used to measure a security's speed (or strength) of
movement.  Or simply the change in price. 

Calculation:
    Default Inputs:
        length: 1
    MOM = close.diff(length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


ppo_docs = \
"""Percentage Price Oscillator (PPO)

The Percentage Price Oscillator is similar to MACD in measuring momentum.

Calculation:
    Default Inputs:
        fast: 12, slow: 26
    SMA = Simple Moving Average
    EMA = Exponential Moving Average
    fast_sma = SMA(close, fast)
    slow_sma = SMA(close, slow)
    PPO = 100 * (fast_sma - slow_sma) / slow_sma
    Signal = EMA(PPO, signal)
    Histogram = PPO - Signal

Sources:
    https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)

Args:
    close(pandas.Series): Series of 'close's
    fast(int): The short period.  Default: 12
    slow(int): The long period.   Default: 26
    signal(int): The signal period.   Default: 9
    offset(int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: ppo, histogram, signal columns
"""


roc_docs = \
"""Rate of Change (ROC)

Rate of Change is an indicator is also referred to as Momentum (yeah, confusingly).
It is a pure momentum oscillator that measures the percent change in price with the
previous price 'n' (or length) periods ago.

Calculation:
    Default Inputs:
        length: 1
    MOM = Momentum
    ROC = 100 * MOM(close, length) / close.shift(length)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
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
        

Returns:
    pd.Series: New feature
"""

t3_docs = \
"""
T3 = c1 * e6 + c2 * e5 + c3 * e4 + c4 * e3
a = 0.7, 0.618 or 0 <= a < 1
c1 = -a^3
c2 = 3a^2 + 3a^3 = 3a^2 * (1 + a)
c3 = -6a^2 - 3a - 3a^3
c4 = a^3 + 3a^2 + 3a + 1

e1 = ema(ts, n)
e2 = ema(e1, n)
e3 = ema(e2, n)
e4 = ema(e3, n)
e5 = ema(e4, n)
e6 = ema(e5, n)
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
        

Returns:
    pd.Series: New feature
"""



# Trend Documentation
adx_docs = \
"""
DMI ADX TREND 2.0 by @TraderR0BERT, NETWORTHIE.COM
    //Created by @TraderR0BERT, NETWORTHIE.COM, last updated 01/26/2016
    //DMI Indicator 
    //Resolution input option for higher/lower time frames

    study(title="DMI ADX TREND 2.0", shorttitle="ADX TREND 2.0")

    adxlen = input(14, title="ADX Smoothing")
    dilen = input(14, title="DI Length")
    thold = input(20, title="Threshold")

    threshold = thold

    //Script for Indicator
    dirmov(len) =>
        up = change(high)
        down = -change(low)
        truerange = rma(tr, len)
        plus = fixnan(100 * rma(up > down and up > 0 ? up : 0, len) / truerange)
        minus = fixnan(100 * rma(down > up and down > 0 ? down : 0, len) / truerange)
        [plus, minus]
        
    adx(dilen, adxlen) => 
        [plus, minus] = dirmov(dilen)
        sum = plus + minus
        adx = 100 * rma(abs(plus - minus) / (sum == 0 ? 1 : sum), adxlen)
        [adx, plus, minus]

    [sig, up, down] = adx(dilen, adxlen)

    osob=input(40,title="Exhaustion Level for ADX, default = 40")


    col = sig >= sig[1] ? green : sig <= sig[1] ? red : gray 

    //Plot Definitions Current Timeframe
    p1 = plot(sig, color=col, linewidth = 3, title="ADX")
    p2 = plot(sig, color=col, style=circles, linewidth=3, title="ADX")
    p3 = plot(up, color=blue, linewidth = 3, title="+DI")
    p4 = plot(up, color=blue, style=circles, linewidth=3, title="+DI")
    p5 = plot(down, color=fuchsia, linewidth = 3, title="-DI")
    p6 = plot(down, color=fuchsia, style=circles, linewidth=3, title="-DI")
    h1 = plot(threshold, color=black, linewidth =3, title="Threshold")


    trender = (sig >= up or sig >= down) ? 1 : 0
    bgcolor(trender>0?black:gray, transp=85)

    //Alert Function for ADX crossing Threshold
    Up_Cross = crossover(up, threshold)
    alertcondition(Up_Cross, title="DMI+ cross", message="DMI+ Crossing Threshold")
    Down_Cross = crossover(down, threshold)
    alertcondition(Down_Cross, title="DMI- cross", message="DMI- Crossing Threshold")
"""

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
        

Returns:
    pd.Series: New feature
"""


massi_docs = \
"""Mass Index (MASSI)

The Mass Index is a non-directional volatility indicator that utilitizes the
High-Low Range to identify trend reversals based on range expansions.

Sources:
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:mass_index

Calculation:
    Default Inputs:
        fast: 9, slow: 25
    hl = high - low
    hl_ema1 = EMA(hl, fast)
    hl_ema2 = EMA(hl_ema1, fast)
    hl_ratio = hl_ema1 / hl_ema2
    massi = SUM(hl_ratio, slow)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    fast (int): The short period.  Default: 9
    slow (int): The long period.   Default: 25
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
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

Returns:
    pd.Series: New feature
"""



# Momentum Documentation
ao.__doc__ = ao_docs
apo.__doc__ = apo_docs
macd.__doc__ = macd_docs
mfi.__doc__ = mfi_docs
mom.__doc__ = mom_docs
ppo.__doc__ = ppo_docs
roc.__doc__ = roc_docs
# rsi.__doc__ = rsi_docs
# stoch.__doc__ = stoch_docs
# trix.__doc__ = trix_docs
# tsi.__doc__ = tsi_docs
# uo.__doc__ = uo_docs
# willr.__doc__ = willr_docs


# Overlap Documentation
# hl2.__doc__ = hl2_docs
# hlc3.__doc__ = hlc3_docs
# ohlc4.__doc__ = ohlc4_docs
# median.__doc__ = median_docs
# midpoint.__doc__ = midpoint_docs
# midprice.__doc__ = midprice_docs
# rpn.__doc__ = rpn_docs
# t3.__doc__ = t3_docs

# Performance Documentation
# log_return.__doc__ = log_return_docs
# percent_return.__doc__ = percent_return_docs

# Statistics Documentation
# kurtosis.__doc__ = kurtosis_docs
# mcv.__doc__ = mcv_docs
# quantile.__doc__ = quantile_docs
# skew.__doc__ = skew_docs
# stdev.__doc__ = stdev_docs
# variance.__doc__ = variance_docs

# Trend Documentation
# adx.__doc__ = adx_docs
# decreasing.__doc__ = decreasing_docs
# dpo.__doc__ = dpo_docs
# increasing.__doc__ = increasing_docs

# Volatility Documentation
# atr.__doc__ = atr_docs
# bbands.__doc__ = bbands_docs
# donchian.__doc__ = donchian_docs
# kc.__doc__ = kc_docs
massi.__doc__ = massi_docs
# true_range.__doc__ = true_range_docs

# Volume Documentation
# ad.__doc__ = ad_docs
# cmf.__doc__ = cmf_docs
# efi.__doc__ = efi_docs
# eom.__doc__ = eom_docs
# nvi.__doc__ = nvi_docs
# obv.__doc__ = obv_docs
# pvol.__doc__ = pvol_docs
# pvt.__doc__ = pvt_docs