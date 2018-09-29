# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

from .utils import get_offset, verify_series


def hl2(high:pd.Series, low:pd.Series, offset=None, **kwargs):
    """HL2 of a Pandas Series
    
    Use help(df.ta.hl2) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    offset = get_offset(offset)

    # Calculate Result
    hl2 = 0.5 * (high + low)

    # Offset
    hl2 = hl2.shift(offset)

    # Name & Category
    hl2.name = "HL2"
    hl2.category = 'overlap'

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[hl2.name] = hl2

    return hl2


def hlc3(high:pd.Series, low:pd.Series, close:pd.Series, offset=None, **kwargs):
    """HLC3 of a Pandas Series
    
    Use help(df.ta.hlc3) for specific documentation where 'df' represents
    the DataFrame you are using.
    """
    # Validate Arguments
    high = verify_series(high)
    low = verify_series(low)
    close = verify_series(close)
    offset = get_offset(offset)

    # Calculate Result
    hlc3 = (high + low + close) / 3

    # Offset
    hlc3 = hlc3.shift(offset)

    # Name & Category
    hlc3.name = "HLC3"
    hlc3.category = 'overlap'

    # If 'append', then add it to the df
    if 'append' in kwargs and kwargs['append']:
        df[hlc3.name] = hlc3

    return hlc3