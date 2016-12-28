# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class ReadData(object):

    def __init__(self, filname):
        self.dataframe = None
        self.filename = filname

    def load(self):
        self.dataframe = pd.read_table(filepath_or_buffer=self.filename,
                                       sep=',',
                                       header=0,
                                       index_col=0)
        return self.dataframe
