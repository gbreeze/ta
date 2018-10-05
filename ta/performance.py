# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .utils import get_offset, verify_series


def log_return(close:pd.Series, length=None, cumulative:bool = False, offset:int = None, **kwargs):
    """Indicator: Log Return
    
    Use help(df.ta.log_return) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 1
    offset = get_offset(offset)

    # Calculate Result
    log_return = np.log(close).diff(periods=length)

    if cumulative:
        log_return = log_return.cumsum()

    # Offset
    log_return = log_return.shift(offset)

    # Name & Category
    log_return.name = f"{'CUM' if cumulative else ''}LOGRET_{length}"
    log_return.category = 'performance'

    return log_return


def percent_return(close:pd.Series, length=None, cumulative:bool = False, offset:int = None, **kwargs):
    """Indicator: Percent Return
    
    Use help(df.ta.percent_return) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    close = verify_series(close)
    length = int(length) if length and length > 0 else 1
    offset = get_offset(offset)

    # Calculate Result
    pct_return = close.pct_change(length)

    if cumulative:
        pct_return = pct_return.cumsum()

    # Offset
    pct_return = pct_return.shift(offset)

    # Name & Category
    pct_return.name = f"{'CUM' if cumulative else ''}PCTRET_{length}"
    pct_return.category = 'performance'

    return pct_return