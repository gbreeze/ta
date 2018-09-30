# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .utils import get_offset, verify_series


def log_return(close:pd.Series, length=None, cumulative:bool = False, percent:bool = False, offset:int = None, **kwargs):
    """Log Return of a Pandas Series
    
    Use help(df.ta.log_return) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 1
    offset = get_offset(offset)
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

    return log_return


def percent_return(close:pd.Series, length=None, cumulative:bool = False, percent:bool = False, offset:int = None, **kwargs):
    """Percent Return of a Pandas Series
    
    Use help(df.ta.percent_return) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 1
    offset = get_offset(offset)
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

    return pct_return