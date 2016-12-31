# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from feature_preprocessing import FeaturePreprocess
import minepy
from scipy.stats import pearsonr

class FeatureEngin(FeaturePreprocess):

    def __init__(self, filename, ispred=False):
        super(FeatureEngin, self).__init__(filename=filename,
                                           ispred=ispred)
        self.run_preprocessor()
        dataframe = self.dataframe.copy()
        self.y = dataframe['SalePrice']
        self.X = dataframe.drop('SalePrice',
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
        self.X = X
        colnames = X.columns
        values = X.values.T.tolist()
        length = X.shape[1]
        rs = np.corrcoef(values) ** 2
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
        colnames = np.array(self.X.copy().columns)
        colnames_keep = colnames[keep]
        self.colnames_keep = colnames_keep
        self.X = self.X[colnames_keep]

    def _domain_knwl_intrplt(self):
        pass

    def _domain_knwl_encod(self):
        X = self.X
        X['LotShap'] = X['LotShap'].map({'Reg': 4, 'IR1': 3,
                                         'IR2': 2, 'IR3':1})
        X['Utilities'] = X['Utilities'].map({'AllPub': 4, 'NoSewr': 3,
                                             'NoSeWa': 2, 'ELO': 1})
        X['LandSlope'] = X['LandSlope'].map({'Gtl': 3, 'Mod':2, 'Sev': 1})
        X['BldgType'] = X['BldgType'].map({'1Fam':5, '2FmCon':4, 'Duplx':3,
                                           'TwnhsE': 2, 'TwnhsI':1})
        X['HouseStyle'] = X['HouseStyle'].map({'SLvl':6, 'SFoyer':5,
                                               '2.5Fin':4, '2.5Unf': 3.5,
                                               '2Story':3, '1.5Fin': 2,
                                               '1.5Unf': 1.5, '1Story': 1})


    def start(self):
        self._remove_collinearity_()