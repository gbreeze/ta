# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

from pandas.core.base import PandasObject

class BasePandasObject(PandasObject):
    def __init__(self, data, **kwargs):
        if data.empty:
            return None
        
        if len(data.columns):
            self._data = data
        else:
            raise AttributeError(f"[X] Not enough columns!")
    
    def __call__(self, kind, *args, **kwargs):
        raise NotImplementedError()
