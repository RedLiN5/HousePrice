# -*- coding: utf-8 -*-
import numpy as np
from feature_engineering import FeatureEngin
import matplotlib.pyplot as plt


class Plots(FeatureEngin):

    def __init__(self):
        super(Plots, self).__init__()

    def feature_importance(self):
        corr_scores = self._corr_()
        x_names = np.array(corr_scores.index)
        scores = corr_scores.values
        x_range = np.arange(1, len(x_names) + 1)
        fig = plt.figure(figsize=(20, 9))
        ax = fig.add_subplot(111)
        ax.plot(x_range,
                scores,
                'c.-')
        ax.set_xticklabels(x_names,
                           rotation=80)
        ax.set_xticks(np.arange(len(x_names)))
        ax.set_ylim([-0.2, 1])
        ax.title.set_text('Feature Importance')
        fig.savefig('myplot.png',
                    bbox_inches='tight')