import matplotlib.pyplot as plt
from feature_preprocessing import FeaturePreprocess
import pandas as pd
import numpy as np

reader = FeaturePreprocess('train.csv', ispred=True)
df = reader.run_preprocessor()

print(df.columns)
y = np.array(df['SalePrice'])

features_name = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea']
n = len(features_name)
figure, axes = plt.subplots(n, 1)

for i, feature in enumerate(features_name):
    axes[i].scatter(df[feature], y,
                    facecolor = 'c')

plt.savefig('feature_effect_plots.png')
