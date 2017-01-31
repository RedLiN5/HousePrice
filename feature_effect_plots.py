import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from feature_preprocessing import FeaturePreprocess
import pandas as pd
import numpy as np

reader = FeaturePreprocess('train.csv', ispred=False)
df = reader.run_preprocessor()

print(len(df.columns))
y = np.array(df['SalePrice'])

features_name = ['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea',
                 'TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd',
                 'BuiltAge', 'RemodelAge', 'MasVnrArea', 'Fireplaces']

# features_name = df.columns[61:]
n = len(features_name)
figure, axes = plt.subplots(n, 1)
color = (0.5529411764705883, 0.6274509803921569, 0.796078431372549)

figure.set_size_inches(10, 50)

for i, feature in enumerate(features_name):
    axes[i].scatter(df[feature], y,
                    facecolor = color,
                    s = 15)
    axes[i].set_xlabel(feature)
    axes[i].set_yticks(np.linspace(-1*10**5, 9*10**5, 6))
    axes[i].yaxis.set_major_formatter(
        mtick.FormatStrFormatter('%.1e'))

figure.text(0.01, 0.5, 'SalePrice', va='center', rotation='vertical')
plt.subplots_adjust(left=0.15, bottom=0.01,
                    right=0.9, top=0.99,
                    wspace=None, hspace=0.5)

plt.savefig('feature_effect_plots.png')
