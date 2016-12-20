# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class ReadData(object):

    def __init__(self):
        self.dataframe = None

    def run(self):
        self.dataframe = pd.read_table(filepath_or_buffer='train.csv',
                                       sep=',',
                                       header=0,
                                       index_col=0)
        return self.dataframe
