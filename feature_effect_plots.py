import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from feature_preprocessing import FeaturePreprocess
import pandas as pd
import numpy as np

reader = FeaturePreprocess('train.csv', ispred=False)
df = reader.run_preprocessor()

y = np.array(df['SalePrice'])

features_name = df.columns

for i in range(len(features_name)):
    plt.scatter(x = df[features_name[i]],
                y = y)
    plt.xlabel(features_name[i])
    plt.ylabel('SalePrice')
    plt.savefig('Feature_effects_plots/{0}.png'.format(features_name[i]))
    plt.close()

