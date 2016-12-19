# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class ReadData(object):

    def __init__(self, filename, key):
        self.filename = filename
        self.key = key
        self.dataframe = None

    def run(self):
        with pd.HDFStore("train.h5", "r") as train:
            self.dataframe = train.get('train')


