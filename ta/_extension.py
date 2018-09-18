# -*- coding: utf-8 -*-
import math
import numpy as np
import pandas as pd

from pandas.core.base import PandasObject


class BasePandasObject(PandasObject):
    """ """
    def __init__(self, data, **kwargs):
        if data.empty:
            return None
        
        if len(data.columns):
            self._data = data
        else:
            raise AttributeError(f"[X] Not enough columns!")
    
    def __call__(self, kind, *args, **kwargs):
        raise NotImplementedError()


@pd.api.extensions.register_dataframe_accessor('ta')
class AnalysisIndicators(BasePandasObject):
    """ """

    def __call__(self, kind=None, **kwargs):
        try:
            indicator = getattr(self, kind.lower())
        except AttributeError:
            raise ValueError(f"kind='{kind.lower()}' is not valid for {self.__class__.__name__}")
        
        return indicator(**kwargs)
    
    def _valid_df(self, name=None):
        try:
            df = self._data
        except AttributeError as ex:
            msg = f"[X] {ex}: Invalid DataFrame"
            if name:
                print(msg + f": {name}")
            else:
                print(msg)
            return None
        return df
        