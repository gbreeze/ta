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

Sources:
    https://www.tradingview.com/wiki/Awesome_Oscillator_(AO)
    https://www.ifcm.co.uk/ntx-indicators/awesome-oscillator

Calculation:
    Default Inputs:
        fast=5, slow=34
    SMA = Simple Moving Average
    median = (high + low) / 2
    AO = SMA(median, fast) - SMA(median, slow)

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

Sources:
    https://www.investopedia.com/terms/p/ppo.asp

Calculation:
    Default Inputs:
        fast=12, slow=26
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

Sources:
    http://www.worden.com/TeleChartHelp/Content/Indicators/Balance_of_Power.htm

Calculation:
    BOP = (close - open) / (high - low)

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
    Default Inputs:
        length=20, c=0.015
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

Sources:
    https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)

Calculation:
    Default Inputs:
        fast=12, slow=26, signal=9
    EMA = Exponential Moving Average
    MACD = EMA(close, fast) - EMA(close, slow)
    Signal = EMA(MACD, signal)
    Histogram = MACD - Signal

Args:
    close (pd.Series): Series of 'close's
    fast (int): The short period.  Default: 12
    slow (int): The long period.   Default: 26
    signal (int): The signal period.   Default: 9
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: macd, histogram, signal columns.
"""


mfi_docs = \
"""Money Flow Index (MFI)

Money Flow Index is an oscillator indicator that is used to measure buying and
selling pressure by utilizing both price and volume.

Sources:
    https://www.tradingview.com/wiki/Money_Flow_(MFI)

Calculation:
    Default Inputs:
        length=14, drift=1
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

Sources:
    http://www.onlinetradingconcepts.com/TechnicalAnalysis/Momentum.html

Calculation:
    Default Inputs:
        length=1
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

Sources:
    https://www.tradingview.com/wiki/MACD_(Moving_Average_Convergence/Divergence)

Calculation:
    Default Inputs:
        fast=12, slow=26
    SMA = Simple Moving Average
    EMA = Exponential Moving Average
    fast_sma = SMA(close, fast)
    slow_sma = SMA(close, slow)
    PPO = 100 * (fast_sma - slow_sma) / slow_sma
    Signal = EMA(PPO, signal)
    Histogram = PPO - Signal

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

Sources:
    https://www.tradingview.com/wiki/Rate_of_Change_(ROC)

Calculation:
    Default Inputs:
        length=1
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
"""Relative Strength Index (RSI)

The Relative Strength Index is popular momentum oscillator used to measure the
velocity as well as the magnitude of directional price movements.

Sources:
    https://www.tradingview.com/wiki/Relative_Strength_Index_(RSI)

Calculation:
    Default Inputs:
        length=14, drift=1
    ABS = Absolute Value
    EMA = Exponential Moving Average
    positive = close if close.diff(drift) > 0 else 0
    negative = close if close.diff(drift) < 0 else 0
    pos_avg = EMA(positive, length)
    neg_avg = ABS(EMA(negative, length))
    RSI = 100 * pos_avg / (pos_avg + neg_avg)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 1
    drift (int): The difference period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


stoch_docs = \
"""Stochastic (STOCH)

Stochastic Oscillator is a range bound momentum indicator.  It displays the location
of the close relative to the high-low range over a period.

Sources:
    https://www.tradingview.com/wiki/Stochastic_(STOCH)

Calculation:
    Default Inputs:
        fast_k=14, slow_k=5, slow_d=3
    SMA = Simple Moving Average
    lowest_low   = low for last fast_k periods
    highest_high = high for last fast_k periods

    FASTK = 100 * (close - lowest_low) / (highest_high - lowest_low)
    FASTD = SMA(FASTK, slow_d)

    SLOWK = SMA(FASTK, slow_k)
    SLOWD = SMA(SLOWK, slow_d)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    fast_k (int): The Fast %K period.  Default: 14
    slow_k (int): The Slow %K period.  Default: 5
    slow_d (int): The Slow %D period.  Default: 3
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: fastk, fastd, slowk, slowd columns.
"""

trix_docs = \
"""Trix (TRIX)

TRIX is a momentum oscillator to identify divergences.

Sources:
    https://www.tradingview.com/wiki/TRIX

Calculation:
    Default Inputs:
        length=18, drift=1
    EMA = Exponential Moving Average
    ROC = Rate of Change
    ema1 = EMA(close, length)
    ema2 = EMA(ema1, length)
    ema3 = EMA(ema2, length)
    TRIX = 100 * ROC(ema3, drift)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 18
    drift (int): The difference period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


tsi_docs = \
"""True Strength Index (TSI)

The True Strength Index is a momentum indicator used to identify short-term
swings while in the direction of the trend as well as determining overbought
and oversold conditions.

Sources:
    https://www.investopedia.com/terms/t/tsi.asp

Calculation:
    Default Inputs:
        fast=13, slow=25, drift=1
    EMA = Exponential Moving Average
    diff = close.diff(drift)

    slow_ema = EMA(diff, slow)
    fast_slow_ema = EMA(slow_ema, slow)

    abs_diff_slow_ema = absolute_diff_ema = EMA(ABS(diff), slow)
    abema = abs_diff_fast_slow_ema = EMA(abs_diff_slow_ema, fast)

    TSI = 100 * fast_slow_ema / abema

Args:
    close (pd.Series): Series of 'close's
    fast (int): The short period.  Default: 13
    slow (int): The long period.   Default: 25
    drift (int): The difference period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


uo_docs = \
"""Ultimate Oscillator (UO)

The Ultimate Oscillator is a momentum indicator over three different
periods.  It attempts to correct false divergence trading signals.

Sources:
    https://www.tradingview.com/wiki/Ultimate_Oscillator_(UO)

Calculation:
    Default Inputs:
        fast=7, medium=14, slow=28, fast_w=4.0, medium_w=2.0, slow_w=1.0, drift=1
    min_low_or_pc  = close.shift(drift).combine(low, min)
    max_high_or_pc = close.shift(drift).combine(high, max)

    bp = buying pressure = close - min_low_or_pc
    tr = true range = max_high_or_pc - min_low_or_pc

    fast_avg = SUM(bp, fast) / SUM(tr, fast)
    medium_avg = SUM(bp, medium) / SUM(tr, medium)
    slow_avg = SUM(bp, slow) / SUM(tr, slow)

    total_weight = fast_w + medium_w + slow_w
    weights = (fast_w * fast_avg) + (medium_w * medium_avg) + (slow_w * slow_avg)
    UO = 100 * weights / total_weight

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    fast (int): The Fast %K period.  Default: 7
    medium (int): The Slow %K period.  Default: 14
    slow (int): The Slow %D period.  Default: 28
    fast_w (float): The Fast %K period.  Default: 4.0
    medium_w (float): The Slow %K period.  Default: 2.0
    slow_w (float): The Slow %D period.  Default: 1.0
    drift (int): The difference period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


willr_docs = \
"""William's Percent R (WILLR)

William's Percent R is a momentum oscillator similar to the RSI that
attempts to identify overbought and oversold conditions.

Sources:
    https://www.tradingview.com/wiki/Williams_%25R_(%25R)

Calculation:
    Default Inputs:
        length=20
    lowest_low   = low.rolling(length).min()
    highest_high = high.rolling(length).max()

    WILLR = 100 * ((close - lowest_low) / (highest_high - lowest_low) - 1)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 14
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


# Trend Documentation
adx_docs = \
"""Average Directional Movement (ADX)

Average Directional Movement is meant to quantify trend strength by measuring
the amount of movement in a single direction.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/average-directional-movement-adx/

Calculation:
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

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 14
    drift (int): The difference period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: adx, dmp, dmn columns.
"""


aroon_docs = \
"""Aroon (AROON)

Aroon attempts to identify if a security is trending and how strong.

Sources:
    https://www.tradingview.com/wiki/Aroon
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/aroon-ar/

Calculation:
    Default Inputs:
        length=1
    def maxidx(x):
        return 100 * (int(np.argmax(x)) + 1) / length

    def minidx(x):
        return 100 * (int(np.argmin(x)) + 1) / length

    _close = close.rolling(length, min_periods=min_periods)
    aroon_up = _close.apply(maxidx, raw=True)
    aroon_down = _close.apply(minidx, raw=True)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: aroon_up, aroon_down columns.
"""


decreasing_docs = \
"""Decreasing

Returns True or False if the series is decreasing.  By default, it returns
True and False as 1 and 0 respectively with kwarg 'asint'.

Sources:

Calculation:
    decreasing = close.diff(length) < 0
    if asint:
        decreasing = decreasing.astype(int)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 1
    asint (bool): Returns as binary.  Default: True
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
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
rsi.__doc__ = rsi_docs
stoch.__doc__ = stoch_docs
trix.__doc__ = trix_docs
tsi.__doc__ = tsi_docs
uo.__doc__ = uo_docs
willr.__doc__ = willr_docs


# Trend Documentation
adx.__doc__ = adx_docs
aroon.__doc__ = aroon_docs
decreasing.__doc__ = decreasing_docs
# dpo.__doc__ = dpo_docs
# ichimoku.__doc__ = ichimoku_docs
# increasing.__doc__ = increasing_docs
# kst.__doc__ = kst_docs
# vortex.__doc__ = vortex_docs

# Volatility Documentation
# atr.__doc__ = atr_docs
# bbands.__doc__ = bbands_docs
# donchian.__doc__ = donchian_docs
# kc.__doc__ = kc_docs
massi.__doc__ = massi_docs
# natr.__doc__ = natr_docs
# true_range.__doc__ = true_range_docs

# Volume Documentation
# ad.__doc__ = ad_docs
# adosc.__doc__ = adosc_docs
# cmf.__doc__ = cmf_docs
# efi.__doc__ = efi_docs
# eom.__doc__ = eom_docs
# nvi.__doc__ = nvi_docs
# obv.__doc__ = obv_docs
# pvol.__doc__ = pvol_docs
# pvt.__doc__ = pvt_docs