# -*- coding: utf-8 -*-

from feature_engineering import FeatureEngin
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from xgboost import XGBRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score
from collections import Counter


class Regressions(FeatureEngin):

    def __init__(self, filename, ispred=False):
        super(Regressions, self).__init__(filename=filename,
                                          ispred=ispred)
        self.feature_names = FeatureEngin.start(self)
        self.predictions = []
        self.ensemble_size = 10
        self.ensemble_models = []
        self._split_data()

    def _split_data(self):
        X, y = self.X.copy(), self.y.copy()
        self.X_train, self.X_test, self.y_train,\
        self.y_test = train_test_split(X, y,
                                       test_size=0.3,
                                       random_state=0)

    def _ridge_reg(self): # 0.736
        clf = Ridge(alpha=0.7,
                    fit_intercept=True,
                    normalize=True,
                    solver='auto',
                    random_state=1,
                    tol=.001)
        self.ensemble_models.append(clf)
        clf.fit(self.X_train.copy(),
                self.y_train.copy())
        pred = clf.predict(self.X_test.copy())
        self.predictions.append(pred)

    def fit_ridge(self, X_test):
        clf = Ridge(alpha=0.7,
                    fit_intercept=True,
                    normalize=True,
                    solver='auto',
                    random_state=1,
                    tol=.001)
        clf.fit(self.X.copy(),
                self.y.copy())
        preds = clf.predict(X=X_test)
        ids = X_test.index
        pred_df = pd.DataFrame({'SalePrice':preds},
                               index=ids)
        pred_df.to_csv('results_ridge.csv',
                       sep=',')

    def _lasso_reg(self): # 0.730
        clf = Lasso(alpha=120,
                    fit_intercept=True,
                    normalize=True,
                    random_state=1)
        self.ensemble_models.append(clf)
        clf.fit(self.X_train.copy(),
                self.y_train.copy())
        pred = clf.predict(self.X_test.copy())
        self.predictions.append(pred)

    def fit_lasso(self, X_test):
        clf = Lasso(alpha=120,
                    fit_intercept=True,
                    normalize=True,
                    random_state=1)
        clf.fit(self.X.copy(),
                self.y.copy())
        preds = clf.predict(X=X_test)
        ids = X_test.index
        pred_df = pd.DataFrame({'SalePrice':preds},
                               index=ids)
        pred_df.to_csv('results_lasso.csv',
                       sep=',')

    def _xgb1_reg(self): # 0.898
        clf = XGBRegressor(max_depth=7,
                           learning_rate=0.11,
                           n_estimators=300,
                           gamma=1,
                           seed=1)
        self.ensemble_models.append(clf)
        clf.fit(self.X_train.copy().values,
                self.y_train.copy().values.flatten())
        pred = clf.predict(self.X_test.copy().values)
        self.predictions.append(pred)

    def _xgb2_reg(self): # 0.898
        clf = XGBRegressor(max_depth=6,
                           learning_rate=0.09,
                           n_estimators=300,
                           gamma=1,
                           seed=1)
        self.ensemble_models.append(clf)
        clf.fit(self.X_train.copy().values,
                self.y_train.copy().values.flatten())
        pred = clf.predict(self.X_test.copy().values)
        self.predictions.append(pred)

    def _xgb_test_reg(self): # 0.898
        depths = [5,6,7,8,9]
        rates = np.linspace(0.09, 0.15, 7)
        ns = [280, 290, 300, 310, 320]
        scores = []
        for depth in depths:
            for rate in rates:
                for n in ns:
                    clf = XGBRegressor(max_depth=depth,
                                       learning_rate=rate,
                                       n_estimators=n,
                                       gamma=1,
                                       seed=1)
                    self.ensemble_models.append(clf)
                    clf.fit(self.X_train.copy().values,
                            self.y_train.copy().values.flatten())
                    score = clf.score(self.X_test.copy().values,self.y_test.copy().values.flatten())
                    scores.append(score)
                    print('depth:', depth, 'rate:', rate, 'n:', n, 'score:', score, '\n')
        print('best score:', sorted(scores, reverse=True)[:3])
        # pred = clf.predict(self.X_test.copy().values)
        # self.predictions.append(pred)

    def fit_xgb(self, X_test):
        clf = XGBRegressor(max_depth=7,
                           learning_rate=0.11,
                           n_estimators=300,
                           gamma=1,
                           seed=1)
        clf.fit(self.X.copy().values,
                self.y.copy().values.flatten())
        preds = clf.predict(data=X_test.values)
        ids = X_test.index
        pred_df = pd.DataFrame({'SalePrice': preds},
                               index=ids)
        pred_df.to_csv('results_xgb.csv',
                       sep=',')

    def _ridge_interaction(self): # 0.749
        model = make_pipeline(PolynomialFeatures(degree=1,
                                                 interaction_only=True),
                              Ridge(alpha=0.75,
                                    fit_intercept=True,
                                    normalize=True,
                                    solver='auto',
                                    random_state=1,
                                    tol=.001))
        self.ensemble_models.append(model)
        model.fit(self.X_train.copy(),
                  self.y_train.copy())
        pred = model.predict(self.X_test.copy())
        self.predictions.append(pred)

    def _lasso_interaction(self): # 0.742
        model = make_pipeline(PolynomialFeatures(degree=1,
                                                 interaction_only=True),
                              Lasso(alpha=170,
                                    fit_intercept=True,
                                    normalize=True,
                                    random_state=1))
        self.ensemble_models.append(model)
        model.fit(self.X_train.copy(),
                  self.y_train.copy())
        pred = model.predict(self.X_test.copy())
        self.predictions.append(pred)

    def fit_polynomial(self):
        pass

    def _generate_weight_indices(self):
        self._ridge_reg()
        self._xgb1_reg()
        self._xgb2_reg()
        self._lasso_reg()
        self._ridge_interaction()
        self._lasso_interaction()
        predictions = self.predictions.copy()
        ensemble = []
        indices = []

        if len(predictions) > 1:
            for i in range(self.ensemble_size):
                scores = np.zeros(len(predictions))
                s = len(ensemble)
                if s == 0:
                    y_weighted_ensemble_pred = np.zeros(predictions[0].shape)
                else:
                    y_ensemble_pred = np.mean(np.array(ensemble),
                                              axis=0)
                    y_weighted_ensemble_pred = (s / float(s + 1)) * \
                                               y_ensemble_pred
                for j, pred in enumerate(predictions):
                    y_comb_ensemble_pred = y_weighted_ensemble_pred +\
                                           1 / (s + 1) * pred
                    scores[j] = r2_score(self.y_test.copy().values.flatten(),
                                         y_comb_ensemble_pred)
                best = np.nanargmax(scores)
                ensemble.append(predictions[best])
                indices.append(best)
        elif len(predictions) == 1:
            pass
        else:
            raise ValueError('No prediction given by models.')
        return indices

    def _calculate_weights(self):
        indices = self._generate_weight_indices()
        weights = np.zeros(len(self.predictions),
                           dtype=float)
        ensemble_members = Counter(indices).most_common()
        if len(indices) > 1:
            for member in ensemble_members:
                weight = member[1] / self.ensemble_size
                weights[member[0]] = weight
            if np.sum(weights) < 1:
                weights = weights / np.sum(weights)
        else:
            pass
        return weights

    def fit_ensemble(self):
        self.weights = self._calculate_weights()
        for i, clf in enumerate(self.ensemble_models):
            clf.fit(self.X.copy().values,
                    self.y.copy().values.flatten())

    def predict_ensemble(self, X_test):
        """
        Predict results with ensemble models.
        :param
        X_test: pandas.DataFrame

        :return:
        """
        weights = self.weights
        weights = [0, 0.38, 0.38, 0, 0.12, 0.12]
        ensemble_results = []
        for i, clf in enumerate(self.ensemble_models):
            pred = clf.predict(X_test.values)
            ensemble_results.append(pred)

        ensemble_results = np.array(ensemble_results)
        result = np.dot(ensemble_results.T,
                        weights)
        ids = X_test.index
        pred_df = pd.DataFrame({'SalePrice': result},
                               index=ids)
        pred_df.to_csv('results_ensemble.csv',
                       sep=',')

    def run(self):
        # TODO(Leslie) add last function to run
        pass
