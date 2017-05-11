from feature_preprocessing import FeaturePreprocess
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import make_scorer
from math import sqrt

def rmse_calculator(y_true, y_pred):
    rmse = sqrt(mean_squared_error(y_true, y_pred))
    return rmse


rmse_scorer = make_scorer(rmse_calculator)


preprocessing = FeaturePreprocess(trainFile='train.csv',
                                  testFile='test.csv')
trainIndex = preprocessing.trainIndex
testIndex = preprocessing.testIndex
y_train = preprocessing.y_train
y_train_log1p = np.log1p(y_train)

df_all = preprocessing._domain_knwl_encoding()
X_train = df_all.loc[trainIndex]
X_test = df_all.loc[testIndex]


reg = xgb.XGBRegressor()

scores = cross_val_score(reg, X_train, y_train_log1p, cv=5, scoring=rmse_scorer)

print(scores)
print("Accuracy: {0:0.6f} (+/- {1:0.6f})".format(np.mean(scores),
                                           np.std(scores)))

# y_test_log1p = reg.predict(X_test)
# y_test = np.exp(y_test_log1p)-1
#
# submit = pd.DataFrame({'Id': testIndex,
#                        'SalePrice': y_test})
# submit.to_csv('results/submit_xgb.csv',
#               index=False)



