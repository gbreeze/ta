# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd


def log_return(close, length=None, cumulative:bool = False, percent:bool = False, offset:int = None, **kwargs):
    # Validate Arguments
    length = int(length) if length and length > 0 else 1
    # min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = offset if isinstance(offset, int) else 0
    percent = 100 if percent else 1

    # Calculate Result
    log_return = percent * np.log(close).diff(periods=length)

    if cumulative:
        log_return = log_return.cumsum()

    # Offset
    log_return = log_return.shift(offset)

    # Name & Category
    log_return.name = f"{'CUM_' if cumulative else ''}LOGRET_{length}"
    log_return.category = 'performance'

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[log_return.name] = log_return

    return log_return


def percent_return(close, length=None, cumulative:bool = False, percent:bool = False, offset:int = None, **kwargs):
    # Validate Arguments
    length = int(length) if length and length > 0 else 1
    # min_periods = int(kwargs['min_periods']) if 'min_periods' in kwargs and kwargs['min_periods'] is not None else length
    offset = offset if isinstance(offset, int) else 0
    percent = 100 if percent else 1

    # Calculate Result
    pct_return = percent * close.pct_change(length)

    if cumulative:
        pct_return = percent * pct_return.cumsum()

    # Offset
    pct_return = pct_return.shift(offset)

    # Name & Category
    pct_return.name = f"{'CUM_' if cumulative else ''}PCTRET_{length}"
    pct_return.category = 'performance'

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[pct_return.name] = pct_return

    return pct_return