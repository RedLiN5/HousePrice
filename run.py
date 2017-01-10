from regressions import Regressions
from feature_preprocessing import FeaturePreprocess
import time

start = time.time()
# *******************************
reg = Regressions('train.csv',
                  ispred=True)
colnames_keep = reg.colnames_keep

test_prep = FeaturePreprocess(filename='test.csv',
                              ispred=True)
test_prep.run_preprocessor()
test_df = test_prep.dataframe
X_test = test_df[colnames_keep]

reg.fit_ensemble()
reg.predict_ensemble(X_test = X_test)
# *******************************


print('Runnning time is %.3f Sec'%(time.time()-start))
