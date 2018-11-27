# Technical Analysis Library in Python

<!-- ![alt text](https://raw.githubusercontent.com/bukosabino/ta/master/doc/figure.png) -->

Technical Analysis (TA) is an easy to use library that is built upon Python's Pandas library with more than 60 Indicators.  These indicators are comminly used for financial time series datasets with columns or labels similar to: datetime, open, high, low, close, volume, et al.  Many commonly used indicators are included, such as: _Moving Average Convergence Divergence_ (*MACD*), _Hull Exponential Moving Average_ (*HMA*), _Bollinger Bands_ (*BBANDS*), _On-Balance Volume_ (*OBV*), _Aroon Oscillator_ (*AROON*) and more.

This version contains both the orignal code branch as well as a newly refactored branch with the option to use [Pandas DataFrame Extension](https://pandas.pydata.org/pandas-docs/stable/extending.html) mode. 
All the indicators return a named Series or a DataFrame in uppercase underscore parameter format.  For example, MACD(fast=12, slow=26, signal=9) will return a DataFrame with columns: ['MACD_12_26_9', 'MACDH_12_26_9', 'MACDS_12_26_9'].


## **Quick Start** using the DataFrame Extension

```python
import pandas as pd
import ta as ta

# Load data
df = pd.read_csv('symbol.csv', sep=',')

# Process and append your indicators to the 'df'
df.ta.ema(length=8, append=True)
df.ta.ema(length=21, append=True)
df.ta.sma(length=50, append=True)
df.ta.sma(length=200, append=True)
df.ta.kc(append=True)

df.ta.macd(fast=8, slow=21, signal=9, append=True)
df.ta.rsi(length=14, append=True)

df.ta.obv(append=True)
df.ta.log_return(cumulative=True, append=True)

# New Columns with results
df.columns

# Take a peek
df.tail()

# vv Continue Post Processing vv
```

## New Changes

* At 70+ indicators.
* Abbreviated Indicator names as listed below.
* *Extended Pandas DataFrame* as 'ta'.  See examples below.
* Parameter names are more consistent.
* Former indicators still exist and are renamed with '_depreciated' append to it's name.  For example, 'average_true_range' is now 'average_true_range_depreciated'.
* Refactoring indicators into categories similar to [TA-lib](https://github.com/mrjbq7/ta-lib/tree/master/docs/func_groups).


### What is a Pandas DataFrame Extension?

A [Pandas DataFrame Extension](https://pandas.pydata.org/pandas-docs/stable/extending.html), extends a DataFrame allowing one to add more functionality and features to Pandas to suit your needs.  As such, it is now easier to run Technical Analysis on existing Financial Time Series without leaving the current DataFrame.  This extension by default returns the Indicator result or, inclusively, it can append the result to the existing DataFrame by including the parameter 
'append=True' in the method call. See examples below.



# Technical Analysis Indicators (by Category)

## _Momentum_ (17)

* _Awesome Oscillator_: **ao**
* _Absolute Price Oscillator_: **apo**
* _Balance of Power_: **bop**
* _Commodity Channel Index_: **cci**
* _Chande Momentum Oscillator_: **cmo**
* _Coppock Curve_: **copc**
* _KST Oscillator_: **kst**
* _Moving Average Convergence Divergence_: **macd**
* _Momentum_: **mom**
* _Percentage Price Oscillator_: **ppo**
* _Rate of Change_: **roc**
* _Relative Strength Index_: **rsi**
* _Stochastic Oscillator_: **stoch**
* _Trix_: **trix**
* _True strength index_: **tsi**
* _Ultimate Oscillator_: **uo**
* _Williams %R_: **willr**


| _Moving Average Convergence Divergence_ (MACD) |
|:--------:|
| ![Example MACD](/doc/Example_SPY_MACD.png) |


## _Overlap_ (18)

* _Double Exponential Moving Average_: **dema**
* _Exponential Moving Average_: **ema**
* _High-Low Average_: **hl2**
* _High-Low-Close Average_: **hlc3**
    * Commonly known as 'Typical Price' in Technical Analysis literature
* _Hull Exponential Moving Average_: **hma**
* _Ichimoku Kinkō Hyō_: **ichimoku**
    * Use: help(ta.ichimoku). Returns two DataFrames.
* _Midpoint_: **midpoint**
* _Midprice_: **midprice**
* _Open-High-Low-Close Average_: **ohlc4**
* _Pascal's Weighted Moving Average_: **pwma**
* _William's Moving Average_: **rma**
* _Simple Moving Average_: **sma**
* _T3 Moving Average_: **t3**
* _Triple Exponential Moving Average_: **tema**
* _Triangular Moving Average_: **trima**
* _Volume Weighted Average Price_: **vwap**
* _Volume Weighted Moving Average_: **vwma**
* _Weighted Moving Average_: **wma**

| _Simple Moving Averages_ (SMA) and _Bollinger Bands_ (BBANDS) |
|:--------:|
| ![Example Chart](/doc/Example_TA_Chart.png) |


## _Performance_ (2)

Use parameter: cumulative=**True** for cumulative results.

* _Log Return_: **log_return**
* _Percent Return_: **percent_return**

| _Percent Return_ (Cumulative) with _Simple Moving Average_ (SMA) |
|:--------:|
| ![Example Cumulative Percent Return](/doc/Example_SPY_CumulativePercentReturn.png) |


## _Statistics_ (9)

* _Kurtosis_: **kurtosis**
* _Mean Absolute Deviation_: **mad**
* _Mean_: **mean**
    * Alias of **sma**
* _Median_: **median**
* _Quantile_: **quantile**
* _Skew_: **skew**
* _Standard Deviation_: **stdev**
* _Variance_: **variance**
* _Z Score_: **zscore**

| _Z Score_ |
|:--------:|
| ![Example Z Score](/doc/Example_SPY_ZScore.png) |


## _Trend_ (6)

* _Average Directional Movement Index_: **adx**
* _Aroon Oscillator_: **aroon**
* _Decreasing_: **decreasing**
* _Detrended Price Oscillator_: **dpo**
* _Increasing_: **increasing**
* _Vortex Indicator_: **vortex**

| _Average Directional Movement Index_ (ADX) |
|:--------:|
| ![Example ADX](/doc/Example_SPY_ADX.png) |


## _Volatility_ (8)

* _Acceleration Bands_: **accbands**
* _Average True Range_: **atr**
* _Bollinger Bands_: **bbands**
* _Donchian Channel_: **donchain**
* _Keltner Channel_: **kc**
* _Mass Index_: **massi**
* _Normalized Average True Range_: **natr**
* _True Range_: **true_range**

| _Average True Range_ (ATR) |
|:--------:|
| ![Example ATR](/doc/Example_SPY_ATR.png) |


## _Volume_ (10)

* _Accumulation/Distribution Index_: **ad**
* _Accumulation/Distribution Oscillator_: **adosc**
* _Chaikin Money Flow_: **cmf**
* _Elder's Force Index_: **efi**
* _Ease of Movement_: **eom**
* _Money Flow Index_: **mfi**
* _Negative Volume Index_: **nvi**
* _On-Balance Volume_: **obv**
* _Price-Volume_: **pvol**
* _Price Volume Trend_: **pvt**

| _On-Balance Volume_ (OBV) |
|:--------:|
| ![Example OBV](/doc/Example_SPY_OBV.png) |


# Documentation

https://technical-analysis-library-in-python.readthedocs.io/en/latest/

# Motivation

* English: https://towardsdatascience.com/technical-analysis-library-to-financial-datasets-with-pandas-python-4b2b390d3543
* Spanish: https://medium.com/datos-y-ciencia/biblioteca-de-an%C3%A1lisis-t%C3%A9cnico-sobre-series-temporales-financieras-para-machine-learning-con-cb28f9427d0

# Python 3 Installation

```sh
$ virtualenv -p python3 virtualenvironment
$ source virtualenvironment/bin/activate
$ pip install ta
```

To use this library you will need a financial time series dataset including “Open”, “High”, “Low”, “Close” and “Volume” columns.  A “Timestamp” or "Date" column is not required, but is typically included anyhow.  It is preferred that the original columns are lowercase, however it will do it's best to find the intended column.  The majority of Technical Analysis Indicators use price or volume.

You should clean or fill NaN values in your dataset before adding technical analysis indicators.

You can get code examples in [examples_to_use](https://github.com/bukosabino/ta/tree/master/examples_to_use) folder.

You can visualize the features in [this notebook](https://github.com/bukosabino/ta/blob/master/examples_to_use/visualize_features.ipynb).

# Getting Started and Examples

## Module and Indicator Help

```python
import pandas as pd
from ta import ta

# Help about this, 'ta', extension
help(pd.DataFrame().ta)

# List of all indicators
pd.DataFrame().ta.indicators()

# Help about the OBV indicator
help(ta.obv)

# Help about the OBV indicator as a DataFrame Extension
help(pd.DataFrame().ta.obv)
```

## Calling an Indicator

```python
# Load data
spy = pd.read_csv('SPY.csv', sep=',')

# Typical Call
spy_ema50 = ta.ema(spy['close'], length=50)

# Extended Call
spy_ema50 = spy.ta.ema(length=50)

# Extended Call with appending to the 'spy' DataFrame and returning the result
# By default, appending is False
spy_ema50 = spy.ta.ema(length=50, append=True)
# Notice as 'spy_ema50' has been appended to 'spy' DataFrame
spy.columns
```

## Additional ways of calling an Indicator

```python
# You can also use the 'kind' parameter.  Below are equivalent calls.
spy_ema50 = spy.ta(kind='ema', length=50)
spy_ema50 = spy.ta(kind='Ema', length=50)

# Using a non-default series as an input.
# For example instead of having 'ema' using the default 'close' column, have it use the 'open' column instead
spy_ema50_open = spy.ta.ema(close='open', length=50)
```


# Legacy Examples
## Example adding all features

```python
import pandas as pd
from ta import *

# Load data
df = pd.read_csv('your-file.csv', sep=',')

# Clean NaN values
df = utils.dropna(df)

# Add ta features filling NaN values
df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_BTC", fillna=True)
```

## Example adding individual features

```python
import pandas as pd
from ta import *

# Load datas
df = pd.read_csv('your-file.csv', sep=',')

# Clean NaN values
df = utils.dropna(df)

# Add bollinger band high indicator filling NaN values
df['bb_high_indicator'] = bollinger_hband_indicator_depreciated(df["Close"], n=20, ndev=2, fillna=True)

# Add bollinger band low indicator filling NaN values
df['bb_low_indicator'] = bollinger_lband_indicator_depreciated(df["Close"], n=20, ndev=2, fillna=True)
```


# Developer Edition

```sh
$ git clone https://github.com/bukosabino/ta.git
$ cd ta
$ pip install -r requirements.txt
$ cd dev
$ python bollinger_band_features_example.py
```


# Inspiration:

* https://en.wikipedia.org/wiki/Technical_analysis
* https://pandas.pydata.org
* https://github.com/FreddieWitherden/ta
* https://github.com/femtotrader/pandas_talib


# TODO:

* Add [more technical analysis features](https://en.wikipedia.org/wiki/Technical_analysis).
* Incorporate dask library to parallelize


# Credits:

Developed by Bukosabino at Lecrin Technologies - http://lecrintech.com

Expanded and Extended (via Pandas) by Kevin Johnson - https://github.com/twopirllc

Please leave any comments, feedback, or suggestions.
