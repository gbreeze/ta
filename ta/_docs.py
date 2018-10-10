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
    volume (pd.Series): Series of 'volume's
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
        fast=7, medium=14, slow=28,
        fast_w=4.0, medium_w=2.0, slow_w=1.0, drift=1
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

Returns True or False if the series is decreasing over a periods.  By default,
it returns True and False as 1 and 0 respectively with kwarg 'asint'.

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
"""Detrend Price Oscillator (DPO)

Is an indicator designed to remove trend from price and make it easier to
identify cycles.

Sources:
    http://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:detrended_price_osci

Calculation:
    Default Inputs:
        length=1, centered=True
    SMA = Simple Moving Average
    drift = int(0.5 * length) + 1
    
    DPO = close.shift(drift) - SMA(close, length)
    if centered:
        DPO = DPO.shift(-drift)

Args:
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 1
    centered (bool): Shift the dpo back by int(0.5 * length) + 1.  Default: True
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


ichimoku_docs = \
"""Ichimoku Kinkō Hyō (ichimoku)

Developed Pre WWII as a forecasting model for financial markets.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/ichimoku-ich/

Calculation:
    Default Inputs:
        tenkan=9, kijun=26, senkou=52
    MIDPRICE = Midprice
    TENKAN_SEN = MIDPRICE(high, low, close, length=tenkan)
    KIJUN_SEN = MIDPRICE(high, low, close, length=kijun)
    CHIKOU_SPAN = close.shift(-kijun)

    SPAN_A = 0.5 * (TENKAN_SEN + KIJUN_SEN)
    SPAN_A = SPAN_A.shift(kijun)

    SPAN_B = MIDPRICE(high, low, close, length=senkou)
    SPAN_B = SPAN_B.shift(kijun)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    tenkan (int): Tenkan period.  Default: 9
    kijun (int): Kijun period.  Default: 26
    senkou (int): Senkou period.  Default: 52
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: Two DataFrames.
        For the visible period: spanA, spanB, tenkan_sen, kijun_sen,
            and chikou_span columns
        For the forward looking period: spanA and spanB columns
"""


increasing_docs = \
"""Increasing

Returns True or False if the series is increasing over a periods.  By default,
it returns True and False as 1 and 0 respectively with kwarg 'asint'.

Sources:

Calculation:
    increasing = close.diff(length) > 0
    if asint:
        increasing = increasing.astype(int)

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


kst_docs = \
"""'Know Sure Thing' (KST)

The 'Know Sure Thing' is a momentum based oscillator and based on ROC.

Sources:
    https://www.tradingview.com/wiki/Know_Sure_Thing_(KST)
    https://www.incrediblecharts.com/indicators/kst.php

Calculation:
    Default Inputs:
        roc1=10, roc2=15, roc3=20, roc4=30,
        sma1=10, sma2=10, sma3=10, sma4=15, signal=9, drift=1
    ROC = Rate of Change
    SMA = Simple Moving Average
    rocsma1 = SMA(ROC(close, roc1), sma1)
    rocsma2 = SMA(ROC(close, roc2), sma2)
    rocsma3 = SMA(ROC(close, roc3), sma3)
    rocsma4 = SMA(ROC(close, roc4), sma4)

    KST = 100 * (rocsma1 + 2 * rocsma2 + 3 * rocsma3 + 4 * rocsma4)
    KST_Signal = SMA(KST, signal)

Args:
    close (pd.Series): Series of 'close's
    roc1 (int): ROC 1 period.  Default: 10
    roc2 (int): ROC 2 period.  Default: 15
    roc3 (int): ROC 3 period.  Default: 20
    roc4 (int): ROC 4 period.  Default: 30
    sma1 (int): SMA 1 period.  Default: 10
    sma2 (int): SMA 2 period.  Default: 10
    sma3 (int): SMA 3 period.  Default: 10
    sma4 (int): SMA 4 period.  Default: 15
    signal (int): It's period.  Default: 9
    drift (int): The difference period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: kst and kst_signal columns
"""


vortex_docs = \
"""Vortex

Two oscillators that capture positive and negative trend movement.

Sources:
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:vortex_indicator

Calculation:
    Default Inputs:
        length=14, drift=1
    TR = True Range
    SMA = Simple Moving Average
    tr = TR(high, low, close)
    tr_sum = tr.rolling(length).sum()

    vmp = (high - low.shift(drift)).abs()
    vmn = (low - high.shift(drift)).abs()

    VIP = vmp.rolling(length).sum() / tr_sum
    VIM = vmn.rolling(length).sum() / tr_sum

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): ROC 1 period.  Default: 14
    drift (int): The difference period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: vip and vim columns
"""

# Volatility Documentation
atr_docs = \
"""Average True Range (ATR)

Averge True Range is used to measure volatility, especially
volatility caused by gaps or limit moves.

Sources:
    https://www.tradingview.com/wiki/Average_True_Range_(ATR)

Calculation:
    Default Inputs:
        length=14, drift=1
    SMA = Simple Moving Average
    EMA = Exponential Moving Average
    TR = True Range
    tr = TR(high, low, close, drift)
    if 'ema':
        ATR = EMA(tr, length)
    else:
        ATR = SMA(tr, length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): It's period.  Default: 14
    mamode (str): Two options: None or 'ema'.  Default: 'ema'
    drift (int): The difference period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


bbands_docs = \
"""Bollinger Bands (BBANDS)

A popular volatility indicator.

Sources:
    https://www.tradingview.com/wiki/Bollinger_Bands_(BB)

Calculation:
    Default Inputs:
        length=20, std=2
    EMA = Exponential Moving Average
    SMA = Simple Moving Average
    STDEV = Standard Deviation
    stdev = STDEV(close, length)
    if 'ema':
        MID = EMA(close, length)
    else:
        MID = SMA(close, length)
    
    LOWER = MID - std * stdev
    UPPER = MID + std * stdev

Args:
    close (pd.Series): Series of 'close's
    length (int): The short period.  Default: 20
    std (int): The long period.   Default: 2
    mamode (str): Two options: None or 'ema'.  Default: 'ema'
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: lower, mid, upper columns.
"""


donchian_docs = \
"""Donchian Channels (DC)

Donchian Channels are used to measure volatility, similar to 
Bollinger Bands and Keltner Channels.

Sources:
    https://www.tradingview.com/wiki/Donchian_Channels_(DC)

Calculation:
    Default Inputs:
        length=20
    LOWER = close.rolling(length).min()
    UPPER = close.rolling(length).max()
    MID = 0.5 * (LOWER + UPPER)

Args:
    close (pd.Series): Series of 'close's
    length (int): The short period.  Default: 20
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: lower, mid, upper columns.
"""


kc_docs = \
"""Keltner Channels (KC)

A popular volatility indicator similar to Bollinger Bands and
Donchian Channels.

Sources:
    https://www.tradingview.com/wiki/Keltner_Channels_(KC)

Calculation:
    Default Inputs:
        length=20, scalar=2
    ATR = Average True Range
    EMA = Exponential Moving Average
    SMA = Simple Moving Average
    if 'ema':
        BASIS = EMA(close, length)
        BAND = ATR(high, low, close)
    else:
        hl_range = high - low
        tp = typical_price = hlc3(high, low, close)
        BASIS = SMA(tp, length)
        BAND = SMA(hl_range, length)
    
    LOWER = BASIS - scalar * BAND
    UPPER = BASIS + scalar * BAND

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The short period.  Default: 20
    scalar (float): A positive float to scale the bands.   Default: 2
    mamode (str): Two options: None or 'ema'.  Default: 'ema'
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.DataFrame: lower, basis, upper columns.
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
    EMA = Exponential Moving Average
    hl = high - low
    hl_ema1 = EMA(hl, fast)
    hl_ema2 = EMA(hl_ema1, fast)
    hl_ratio = hl_ema1 / hl_ema2
    MASSI = SUM(hl_ratio, slow)

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


natr_docs = \
"""Normalized Average True Range (NATR)

Normalized Average True Range attempt to normalize the average
true range.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/normalized-average-true-range-natr/

Calculation:
    Default Inputs:
        length=20
    ATR = Average True Range
    NATR = (100 / close) * ATR(high, low, close)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    length (int): The short period.  Default: 20
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature
"""


true_range_docs = \
"""True Range

An method to expand a classical range (high minus low) to include
possible gap scenarios.

Sources:
    https://www.macroption.com/true-range/

Calculation:
    Default Inputs:
        drift=1
    ABS = Absolute Value
    prev_close = close.shift(drift)
    TRUE_RANGE = ABS([high - low, high - prev_close, low - prev_close]) 

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    drift (int): The shift period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature
"""



# Volume Documentation
ad_docs = \
"""Accumulation/Distribution (AD)

Accumulation/Distribution indicator utilizes the relative position
of the close to it's High-Low range with volume.  Then it is cumulated.

Sources:
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/accumulationdistribution-ad/

Calculation:
    CUM = Cumulative Sum
    if 'open':
        AD = close - open
    else:
        AD = 2 * close - high - low

    hl_range = high - low
    AD = AD * volume / hl_range
    AD = CUM(AD)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    open (pd.Series): Series of 'open's
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


adosc_docs = \
"""Accumulation/Distribution Oscillator or Chaikin Oscillator

Accumulation/Distribution Oscillator indicator utilizes 
Accumulation/Distribution and treats it similarily to MACD
or APO.

Sources:
    https://www.investopedia.com/articles/active-trading/031914/understanding-chaikin-oscillator.asp

Calculation:
    Default Inputs:
        fast=12, slow=26
    AD = Accum/Dist
    ad = AD(high, low, close, open)
    fast_ad = EMA(ad, fast)
    slow_ad = EMA(ad, slow)
    ADOSC = fast_ad - slow_ad

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    open (pd.Series): Series of 'open's
    volume (pd.Series): Series of 'volume's
    fast (int): The short period.  Default: 12
    slow (int): The long period.   Default: 26
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


cmf_docs = \
"""Chaikin Money Flow (CMF)

Chailin Money Flow measures the amount of money flow volume over a specific
period in conjunction with Accumulation/Distribution.

Sources:
    https://www.tradingview.com/wiki/Chaikin_Money_Flow_(CMF)
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:chaikin_money_flow_cmf

Calculation:
    Default Inputs:
        length=20
    if 'open':
        ad = close - open
    else:
        ad = 2 * close - high - low
    
    hl_range = high - low
    ad = ad * volume / hl_range
    CMF = SUM(ad, length) / SUM(volume, length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    open (pd.Series): Series of 'open's
    volume (pd.Series): Series of 'volume's
    length (int): The short period.  Default: 20
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


efi_docs = \
"""Elder's Force Index (EFI)

Elder's Force Index measures the power behind a price movement using price
and volume as well as potential reversals and price corrections.

Sources:
    https://www.tradingview.com/wiki/Elder%27s_Force_Index_(EFI)
    https://www.motivewave.com/studies/elders_force_index.htm

Calculation:
    Default Inputs:
        length=20, drift=1, mamode=None
    EMA = Exponential Moving Average
    SMA = Simple Moving Average

    pv_diff = close.diff(drift) * volume
    if mamode == 'sma':
        EFI = SMA(pv_diff, length)
    else:
        EFI = EMA(pv_diff, length)

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): The short period.  Default: 13
    drift (int): The diff period.   Default: 1
    mamode (str): Two options: None or 'sma'.  Default: None
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


eom_docs = \
"""Ease of Movement (EOM)

Ease of Movement is a volume based oscillator that is designed to measure the
relationship between price and volume flucuating across a zero line.

Sources:
    https://www.tradingview.com/wiki/Ease_of_Movement_(EOM)
    https://www.motivewave.com/studies/ease_of_movement.htm
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:ease_of_movement_emv

Calculation:
    Default Inputs:
        length=14, divisor=100000000, drift=1
    SMA = Simple Moving Average    
    hl_range = high - low
    distance = 0.5 * (high - high.shift(drift) + low - low.shift(drift))
    box_ratio = (volume / divisor) / hl_range
    eom = distance / box_ratio
    EOM = SMA(eom, length)

Args:
    high (pd.Series): Series of 'high's
    low (pd.Series): Series of 'low's
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): The short period.  Default: 14
    drift (int): The diff period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


nvi_docs = \
"""Negative Volume Index (NVI)

The Negative Volume Index is a cumulative indicator that uses volume change in
an attempt to identify where smart money is active.

Sources:
    https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:negative_volume_inde
    https://www.motivewave.com/studies/negative_volume_index.htm

Calculation:
    Default Inputs:
        length=20, initial=1000
    ROC = Rate of Change

    roc = ROC(close, length)
    signed_volume = signed_series(volume, initial=1)
    nvi = signed_volume[signed_volume < 0].abs() * roc_
    nvi.fillna(0, inplace=True)
    nvi.iloc[0]= initial
    nvi = nvi.cumsum()

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    length (int): The short period.  Default: 13
    initial (int): The short period.  Default: 1000
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


obv_docs = \
"""On Balance Volume (OBV)

On Balance Volume is a cumulative indicator to measure buying and selling
pressure.

Sources:
    https://www.tradingview.com/wiki/On_Balance_Volume_(OBV)
    https://www.tradingtechnologies.com/help/x-study/technical-indicator-definitions/on-balance-volume-obv/
    https://www.motivewave.com/studies/on_balance_volume.htm

Calculation:
    signed_volume = signed_series(close, initial=1) * volume
    obv = signed_volume.cumsum()

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


pvol_docs = \
"""Price-Volume (PVOL)

Returns a series of the product of price and volume.

Calculation:
    if signed:
        pvol = signed_series(close, 1) * close * volume
    else:
        pvol = close * volume

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    signed (bool): Keeps the sign of the difference in 'close's.  Default: True
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
"""


pvt_docs = \
"""Price-Volume Trend (PVT)

The Price-Volume Trend utilizes the Rate of Change with volume to
and it's cumulative values to determine money flow.

Sources:
    https://www.tradingview.com/wiki/Price_Volume_Trend_(PVT)

Calculation:
    Default Inputs:
        drift=1
    ROC = Rate of Change
    pv = ROC(close, drift) * volume
    PVT = pv.cumsum()

Args:
    close (pd.Series): Series of 'close's
    volume (pd.Series): Series of 'volume's
    drift (int): The diff period.   Default: 1
    offset (int): How many periods to offset the result.  Default: 0

Kwargs:
    fillna (value, optional): pd.DataFrame.fillna(value)
    fill_method (value, optional): Type of fill method

Returns:
    pd.Series: New feature generated.
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
dpo.__doc__ = dpo_docs
ichimoku.__doc__ = ichimoku_docs
increasing.__doc__ = increasing_docs
kst.__doc__ = kst_docs
vortex.__doc__ = vortex_docs

# Volatility Documentation
atr.__doc__ = atr_docs
bbands.__doc__ = bbands_docs
donchian.__doc__ = donchian_docs
kc.__doc__ = kc_docs
massi.__doc__ = massi_docs
natr.__doc__ = natr_docs
true_range.__doc__ = true_range_docs

# Volume Documentation
ad.__doc__ = ad_docs
adosc.__doc__ = adosc_docs
cmf.__doc__ = cmf_docs
efi.__doc__ = efi_docs
eom.__doc__ = eom_docs
nvi.__doc__ = nvi_docs
obv.__doc__ = obv_docs
pvol.__doc__ = pvol_docs
pvt.__doc__ = pvt_docs