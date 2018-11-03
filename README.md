# Technical Analysis Library in Python

<!-- ![alt text](https://raw.githubusercontent.com/bukosabino/ta/master/doc/figure.png) -->

![Example Chart](https://github.com/twopirllc/ta/tree/qt-df-extension/doc/Example_TA_Chart.png)

Technical Analysis is a library, with more than 60 Indicators, to financial time series datasets (open, close, high, low, volume). You can use it to do feature engineering from financial datasets. It is built upon Python Pandas library.

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

Example MACD:
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

## Performance (2)

* Log Return: log_return
* Percent Return: percent_return

Example Cumulative Percent Return:
![Example MACD](https://github.com/twopirllc/ta/tree/qt-df-extension/doc/Example_SPY_CumulativePercentReturn.png)

## Statistics (7)

* Kurtosis: kurtosis
* Median: median
* Quantile: quantile
* Skew: skew
* Standard Deviation: stdev
* Variance: variance
* Z Score: zscore

Example Z Score:
![Example Z Score](https://github.com/twopirllc/ta/tree/qt-df-extension/doc/Example_SPY_ZScore.png)


## Trend (6)

* Average Directional Movement Index: adx
* Aroon Oscillator: aroon
* Decreasing: decreasing
* Detrended Price Oscillator: dpo
* Increasing: increasing
* Vortex Indicator: vortex


## Volatility (7)

* Average True Range: atr
* Bollinger Bands: bbands
* Donchian Channel: donchain
* Keltner Channel: kc
* Mass Index: massi
* Normalized Average True Range: natr
* True Range: true_range


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

#### Example adding all features

```python
import pandas as pd
from ta import *

# Load datas
df = pd.read_csv('your-file.csv', sep=',')

# Clean NaN values
df = utils.dropna(df)

# Add ta features filling NaN values
df = add_all_ta_features(df, "Open", "High", "Low", "Close", "Volume_BTC", fillna=True)
```


#### Example adding individual features

```python
import pandas as pd
from ta import *

# Load datas
df = pd.read_csv('your-file.csv', sep=',')

# Clean NaN values
df = utils.dropna(df)

# Add bollinger band high indicator filling NaN values
df['bb_high_indicator'] = bollinger_hband_indicator(df["Close"], n=20, ndev=2, fillna=True)

# Add bollinger band low indicator filling NaN values
df['bb_low_indicator'] = bollinger_lband_indicator(df["Close"], n=20, ndev=2, fillna=True)
```


# Deploy to developers

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

Please, let us know about any comment or feedback.
