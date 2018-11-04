# Technical Analysis Library in Python

<!-- ![alt text](https://raw.githubusercontent.com/bukosabino/ta/master/doc/figure.png) -->

![Example Chart](https://github.com/twopirllc/ta/tree/qt-df-extension/doc/Example_TA_Chart.png)

Technical Analysis (TA) is a Python library, with more than 60 Indicators, for financial time series datasets (open, close, high, low, volume). You can use it to do feature engineering from financial datasets. It is built upon Python Pandas library.

## New Changes

* Most noteably, doubling the number of indicators.
* Abbreviated Indicator names as listed below.
* *Extended Pandas DataFrame* as 'ta'.  See examples below.
* Parameter names are more consistent.
* Former indicators still exist and are renamed with '_depreciated' append to it's name.  For example, 'average_true_range' is now 'average_true_range_depreciated'.
* Refactoring indicators into categories similar to [TA-lib](https://github.com/mrjbq7/ta-lib/tree/master/docs/func_groups).


### What is Pandas DataFrame Extension?

A [Pandas DataFrame Extension](https://pandas.pydata.org/pandas-docs/stable/extending.html), extends a DataFrame allowing one to add more functionality and features to Pandas to suit your needs.  As such, it is now easier to run Technical Analysis on existing Financial Time Series without leaving the current DataFrame.  This extension by default returns the Indicator result or, inclusively, it can append the result to the existing DataFrame by including the parameter 
'append=True' in the method call. See examples below.


## Momentum (15)

* Awesome Oscillator: ao
* Absolute Price Oscillator: apo
* Balance of Power: bop
* Commodity Channel Index: cci
* KST Oscillator: kst
* Moving Average Convergence Divergence: macd
* Momentum: mom
* Percentage Price Oscillator: ppo
* Rate of Change: roc
* Relative Strength Index: rsi
* Stochastic Oscillator: stoch
* Trix: trix
* True strength index: tsi
* Ultimate Oscillator: uo
* Williams %R: willr

![Example MACD](https://github.com/twopirllc/ta/tree/qt-df-extension/doc/Example_SPY_MACD.png)


## Overlap (17)

* Double Exponential Moving Average: dema
* Exponential Moving Average: ema
* High-Low Average: hl2
* High-Low-Close Average: hlc3
* Hull Exponential Moving Average: hma
* Ichimoku Kinkō Hyō: ichimoku
* Midpoint: midpoint
* Midprice: midprice
* Open-High-Low-Close Average: ohlc4
* William's Moving Average: rma
* Simple Moving Average: sma
* T3 Moving Average: t3
* Triple Exponential Moving Average: tema
* Triangular Moving Average: trima (requires scipy)
* Volume Weighted Average Price: vwap
* Volume Weighted Moving Average: vwma
* Weighted Moving Average: wma

Example: See the first chart above.


## Performance (2)

* Log Return: log_return
* Percent Return: percent_return

![Example Cumulative Percent Return](https://github.com/twopirllc/ta/tree/qt-df-extension/doc/Example_SPY_CumulativePercentReturn.png)


## Statistics (7)

* Kurtosis: kurtosis
* Median: median
* Quantile: quantile
* Skew: skew
* Standard Deviation: stdev
* Variance: variance
* Z Score: zscore

![Example Z Score](https://github.com/twopirllc/ta/tree/qt-df-extension/doc/Example_SPY_ZScore.png)


## Trend (6)

* Average Directional Movement Index: adx
* Aroon Oscillator: aroon
* Decreasing: decreasing
* Detrended Price Oscillator: dpo
* Increasing: increasing
* Vortex Indicator: vortex

![Example ADX](https://github.com/twopirllc/ta/tree/qt-df-extension/doc/Example_SPY_ADX.png)


## Volatility (7)

* Average True Range: atr
* Bollinger Bands: bbands
* Donchian Channel: donchain
* Keltner Channel: kc
* Mass Index: massi
* Normalized Average True Range: natr
* True Range: true_range

![Example ATR](https://github.com/twopirllc/ta/tree/qt-df-extension/doc/Example_SPY_ATR.png)


## Volume (10)

* Accumulation/Distribution Index: ad
* Accumulation/Distribution Oscillator: adosc
* Chaikin Money Flow: cmf
* Elder's Force Index: efi
* Ease of Movement: eom
* Money Flow Index: mfi
* Negative Volume Index: nvi
* On-Balance Volume: obv
* Price-Volume: pvol
* Price Volume Trend: pvt

Example OBV:
![Example OBV](https://github.com/twopirllc/ta/tree/qt-df-extension/doc/Example_SPY_OBV.png)


# Documentation

https://technical-analysis-library-in-python.readthedocs.io/en/latest/

# Motivation

* English: https://towardsdatascience.com/technical-analysis-library-to-financial-datasets-with-pandas-python-4b2b390d3543
* Spanish: https://medium.com/datos-y-ciencia/biblioteca-de-an%C3%A1lisis-t%C3%A9cnico-sobre-series-temporales-financieras-para-machine-learning-con-cb28f9427d0

# How to use (python 3)

```sh
$ virtualenv -p python3 virtualenvironment
$ source virtualenvironment/bin/activate
$ pip install ta
```

To use this library you should have a financial time series dataset including “Timestamp”, “Open”, “High”, “Low”, “Close” and “Volume” columns.

You should clean or fill NaN values in your dataset before adding technical analysis features.

You can get code examples in [examples_to_use](https://github.com/bukosabino/ta/tree/master/examples_to_use) folder.

You can visualize the features in [this notebook](https://github.com/bukosabino/ta/blob/master/examples_to_use/visualize_features.ipynb).

### Indicator Help

```python
import pandas as pd
from ta import ta

# Help about an indicator
help(ta.obv)

# Help about an indicator as a DataFrame Extension
help(pd.DataFrame().ta.obv)
```

### Calling an Indicator

```python
# Download some data from Alpha Vantage
import alphaVantageAPI as av

# Download Daily 'SPY' data
spy = pd.DataFrame().av.D('SPY')

# Typical Call
spy_ema50 = ta.ema(spy['close'], length=50)

# Extended Call
spy_ema50 = spy.ta.ema(length=50)

# Extended Call with appending to the DataFrame and returning the result
# By default, apending is False
spy_ema50 = spy.ta.ema(length=50, append=True)
```

### Additional ways of calling an Indicator

```python
# You can also use the 'kind' parameter.  The 'kind' automatically lowercases 'kind' so either is equivalent
spy_ema50 = spy.ta(kind='ema', length=50)
spy_ema50 = spy.ta(kind='Ema', length=50)

# Using a non-default series as an input.
# For example instead of 'ema' of using the default 'close' column, use 'open' instead
spy_ema50_open = spy.ta.ema(close='open', length=50)
```


## Legacy Examples
### Example adding all features

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

### Example adding individual features

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


## Developer Edition

```sh
$ git clone https://github.com/bukosabino/ta.git
$ cd ta
$ pip install -r requirements.txt
$ cd dev
$ python bollinger_band_features_example.py
```


# Based on:

* https://en.wikipedia.org/wiki/Technical_analysis
* https://pandas.pydata.org
* https://github.com/FreddieWitherden/ta
* https://github.com/femtotrader/pandas_talib


# TODO:

* add [more technical analysis features](https://en.wikipedia.org/wiki/Technical_analysis).
* use dask library to parallelize


# Credits:

Developed by Bukosabino at Lecrin Technologies - http://lecrintech.com

Refactored, Expanded and Extended (via Pandas) by Kevin Johnson - https://github.com/twopirllc

Please leave any comments, feedback, or suggestions.
