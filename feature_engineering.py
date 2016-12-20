# -*- coding: utf-8 -*-

import numpy as np
from datasets import ReadData

class FeatureEngin(ReadData):

    def __init__(self):
        super(FeatureEngin, self).__init__()
        self.run()
        self.rownum = self.dataframe.shape[0]
        self.colnum = self.dataframe.shape[1]

    def _missing_value(self):
        dataframe = self.dataframe
        missing_count = dataframe.isnull().sum()
        return missing_count

    def _remove_missing(self):
        missing_count = self._missing_value()
        col_remove = missing_count[missing_count>self.colnum * .4]
        colname_remove = col_remove.index.tolist()
        self.dataframe = self.dataframe.drop(colname_remove,
                                             axis = 1)

    def _convert(self):
        self._remove_missing()
        colname = self.dataframe.columns
        value_count = [] * self.dataframe.shape[0]
        