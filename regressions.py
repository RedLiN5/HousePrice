# -*- coding: utf-8 -*-

from feature_engineering import FeatureEngin
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

class Regressions(FeatureEngin):

    def __init__(self, filename):
        super(Regressions, self).__init__(filename=filename)
        self.feature_names = FeatureEngin.start(self)

    def _split_data_(self):
        X, y = self.X.copy(), self.y.copy()
        X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                            test_size=0.3,
                                                            random_state=0)
        return X_train, X_test, y_train, y_test

    def ridge_reg(self):
        X_train, X_test, y_train, y_test  = self._split_data_()
        alphas = np.linspace(.1, 3, 20)
        scores = [None] * len(alphas)
        i = 0
        for alpha in alphas:
            clf = Ridge(alpha=alpha,
                        fit_intercept=True,
                        normalize=True,
                        solver='auto',
                        random_state=1,
                        tol=.001)
            clf.fit(X_train, y_train)
            scores[i] = clf.score(X_test, y_test)
            i = i + 1
        print(alphas)
        print(scores)

    def fit_ridge(self, X_test):
        clf = Ridge(alpha=0.7,
                    fit_intercept=True,
                    normalize=True,
                    solver='auto',
                    random_state=1,
                    tol=.001)
        clf.fit(self.X.copy(), self.y.copy())
        preds = clf.predict(X=X_test)
        ids = X_test.index
        pred_df = pd.DataFrame({'SalePrice':preds},
                               index=ids)
        pred_df.to_csv('results_ridge.csv',
                       sep=',')

    def run(self):
        # TODO(Leslie) add last function to run
        pass
