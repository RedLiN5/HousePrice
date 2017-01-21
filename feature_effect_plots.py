import matplotlib.pyplot as plt
from feature_preprocessing import FeaturePreprocess
import pandas as pd
import numpy as np

reader = FeaturePreprocess('train.csv', ispred=True)
df = reader.run_preprocessor()

print(df.columns)
y = np.array(df['SalePrice'])

OverallQual = df['OverallQual'].values.flatten()
GrLivArea = df['GrLivArea']
GarageCars = df['GarageCars']
GarageArea = df['GarageArea']

figure, axes = plt.subplots(4, 1)
axes[0].scatter(OverallQual, y)
axes[1].scatter(GrLivArea, y)
axes[2].scatter(GarageCars, y)
axes[3].scatter(GarageArea, y)
plt.savefig('feature_effect_plots.png')
