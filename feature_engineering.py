# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from feature_preprocessing import FeaturePreprocess
import minepy
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns

class FeatureEngin(FeaturePreprocess):

    def __init__(self):
        super(FeatureEngin, self).__init__()
        self.run_preprocessor()
        self.y = self.dataframe['SalePrice']
        self.X = self.dataframe.drop('SalePrice',
                                     axis = 1)

    def _corr_(self):
        X, y = self.X, self.y
        colnames = X.columns
        m = minepy.MINE()
        corr_scores = pd.Series([None] * len(colnames),
                                index = colnames)
        for colname in colnames:
            x = X[colname]
            m.compute_score(x, y)
            mic = np.around(m.mic(),
                            decimals=4)
            pearson = np.around(pearsonr(x, y)[0],
                                decimals=4)
            corr_scores[colname] = max(mic, pearson)

        corr_scores = corr_scores.sort_values(ascending=False)
        return corr_scores

    def _corr_plot_(self):
        corr_scores = self._corr_()
        x_names = np.array(corr_scores.index)
        scores = corr_scores.values
        x_range = list(range(1, len(x_names) + 1))
        plt.plot(x_range, scores, 'c.-')
        plt.xticks(x_range, x_names,
                   rotation = -45)
        plt.savefig('myplot.png')