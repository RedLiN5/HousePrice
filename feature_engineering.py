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
        self.vif = None

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

    def _vif_calculator_(self):
        """
        Parameters:
        -----------
        X : pands.DataFrame
            DataFrame containing multiple variables and observations for predictors.

        Returns:
        --------
        vif : pandas.DataFrame
            vif values between two predictors.
        """
        feature_names = self._feature_select_()
        X = self.X.copy()
        X = X[feature_names]
        colnames = X.columns
        values = X.values.T
        length = X.shape[1]
        rs = np.corrcoef(values) ** 2
        # AttributeError: 'float' object has no attribute 'sqrt'
        for i in range(length):
            rs[i, i] = 0
        vif_values = 1 / (1 - rs)
        for j in range(length):
            vif_values[j, j:] = 0
        self.vif = pd.DataFrame(data=vif_values,
                                columns=colnames,
                                index=colnames)
        return self.vif

    def _remove_collinearity_(self):
        """
        If VIF score in [1, 5): acceptable
                        [5, 10): problematic
                        [10, inf): disaster
        :return:
        """
        vif = self._vif_calculator_()
        keep = ~(vif >= 5).any().values
        colnames = self.X.copy()
        colnames_keep = np.array(colnames[keep])
        self.X = self.X[colnames_keep]

    def start(self):
        self._remove_collinearity_()