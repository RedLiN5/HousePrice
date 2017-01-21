import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
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
    axes[i].set_xlabel(feature)
    axes[i].set_yticks(np.linspace(-1*10**5, 9*10**5, 6))
    axes[i].yaxis.set_major_formatter(
        mtick.FormatStrFormatter('%.2e'))

figure.text(0, 0.5, 'SalePrice', va='center', rotation='vertical')
plt.subplots_adjust(left=0.15, bottom=0.1,
                    right=0.9, top=0.9,
                    wspace=None, hspace=1)

plt.savefig('feature_effect_plots.png')
