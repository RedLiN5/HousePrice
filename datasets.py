# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

class ReadData(object):

    def __init__(self, trainFile, testFile):
        trainData = pd.read_table(filepath_or_buffer=trainFile,
                                  sep=',',
                                  header=0,
                                  index_col=0)
        testData = pd.read_table(filepath_or_buffer=testFile,
                                 sep=',',
                                 header=0,
                                 index_col=0)
        y = trainData['SalePrice']
        trainData.drop('SalePrice',
                       axis=1,
                       inplace=True)
        self.dataframe = pd.concat([trainData, testData])
        self.trainIndex = self.dataframe.index[:1460]
        self.testIndex = self.dataframe.index[1460:]


