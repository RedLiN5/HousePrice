# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from feature_preprocessing import FeaturePreprocess
import minepy
from scipy.stats import pearsonr

class FeatureEngin(FeaturePreprocess):

    def __init__(self):
        super(FeatureEngin, self).__init__()
        self.run_preprocessor()
        self.y = self.dataframe['SalePrice']
        self.X = self.dataframe.drop('SalePrice',
                                     axis = 1)

    def _feature_impact_(self):
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

    def _feature_select_(self):
        corr_scores = self._feature_impact_()
        score_ratios = corr_scores / sum(corr_scores)
        for i in range(len(score_ratios)):
            while sum(score_ratios) >= .8:
                break

        feature_names = corr_scores.index.tolist()[:i]
        return feature_names

    def _collinearity_handle_(self):
        # TODO(Leslie) Handle collinearity in features
        pass