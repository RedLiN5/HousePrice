from regressions import Regressions
from feature_preprocessing import FeaturePreprocess
import time
import numpy as np
from scipy.stats import skew
from sklearn.preprocessing import StandardScaler

start = time.time()
# *******************************
train_reader= FeaturePreprocess(filename='train.csv',
                                ispred=False)
df = train_reader.run_preprocessor()
X_train = df.drop('SalePrice', axis = 1)
y_train = df['SalePrice']
valid_columns = X_train.columns

numeric_features = train_reader.numeric_features
skewed = X_train[numeric_features].apply(lambda x: skew(x.dropna().astype(float)))

skewed = skewed[abs(skewed) > 0.7]
skewed = skewed.index
X_train[skewed] = np.log1p(X_train[skewed])

scaler = StandardScaler()
scaler.fit(X_train[numeric_features])
scaled = scaler.transform(X_train[numeric_features])
for i, col in enumerate(numeric_features):
    X_train[col] = scaled[:, i]


test_prep = FeaturePreprocess(filename='test.csv',
                              ispred=True)
test_prep.run_preprocessor()
test_df = test_prep.dataframe
X_test = test_df[valid_columns]

X_test[skewed] = np.log1p(X_test[skewed])

scaler = StandardScaler()
scaler.fit(X_test[numeric_features])
scaled = scaler.transform(X_test[numeric_features])
for i, col in enumerate(numeric_features):
    X_test[col] = scaled[:, i]

reg = Regressions(X_train = X_train,
                  y_train = y_train)
reg.fit_ensemble()
reg.predict_ensemble(X_test=X_test)
# *******************************


print('Runnning time is %.3f Sec'%(time.time()-start))
