
import numpy as np
import pandas as pd
import xgboost as xgb
from feature_preprocessing import FeaturePreprocess
from mydecorators import timeit
from hyperopt import hp
from hyperopt import fmin
from hyperopt import tpe
from hyperopt import Trials
from hyperopt import STATUS_OK
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from sklearn.model_selection import cross_val_score
from math import sqrt

def rmse_calculator(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse

rmse_scorer = make_scorer(rmse_calculator)


class HyperparameterOpt(object):

    def __init__(self):
        preprocessing = FeaturePreprocess(trainFile='train.csv',
                                          testFile='test.csv')
        trainIndex = preprocessing.trainIndex
        testIndex = preprocessing.testIndex
        y_train = preprocessing.y_train.reset_index(drop=True)
        self.y_train_log1p = np.log1p(y_train)
        df_all = preprocessing._fill_NA()
        self.X_train = df_all.loc[trainIndex].reset_index(drop=True)
        self.X_test = df_all.loc[testIndex].reset_index(drop=True)

    def _space_xgb(self):
        space = {}
        space['max_depth'] = hp.choice('max_depth',
                                       np.arange(20, 110, 10))
        space['n_estimators'] = hp.choice('n_estimators',
                                          np.arange(200, 1200, 200))
        return space

    def _objective_xgb(self, params):
        reg = xgb.XGBRegressor(**params,
                               gamma=0.1)
        scores = cross_val_score(reg,
                                 self.X_train,
                                 self.y_train_log1p,
                                 cv=5,
                                 scoring=rmse_scorer)
        score_mean = scores.mean()
        return {'loss': -score_mean, 'status': STATUS_OK}

    @timeit
    def optimize_xgb(self):
        trials = Trials()
        space = self._space_xgb()
        best = fmin(self._objective_xgb,
                    space=space,
                    algo=tpe.suggest,
                    max_evals=20,
                    trials=trials)
        print(best)


if __name__ == '__main__':
    hpo = HyperparameterOpt()
    hpo.optimize_xgb()