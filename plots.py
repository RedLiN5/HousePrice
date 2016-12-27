# -*- coding: utf-8 -*-
import numpy as np
from feature_engineering import FeatureEngin
import matplotlib.pyplot as plt


class Plots(FeatureEngin):

    def __init__(self):
        super(Plots, self).__init__()

    def feature_impact(self):
        corr_scores = self._feature_impact_()
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
        ax.set_ylim([-0.2, 1.1])
        ax.title.set_text('Feature Impact')
        fig.savefig('feature_impact.png',
                    bbox_inches='tight')

    def feature_impact_ratio(self):
        corr_scores = self._feature_impact_()
        x_names = np.array(corr_scores.index)
        scores = corr_scores.values
        sum_score = np.round(sum(scores), 2)
        score_ratios = scores/sum_score
        length = len(score_ratios)
        recur_ratios = [None] * length
        recur_ratios[0] = score_ratios[0]
        for i in range(1, length):
            recur_ratios[i] = recur_ratios[i-1] \
                              + score_ratios[i]

        try:
            valid = sum(recur_ratios)
        except TypeError as e:
            raise TypeError(e)

        x_range = np.arange(1, length + 1)
        fig = plt.figure(figsize=(20, 9))
        ax = fig.add_subplot(111)
        ax.plot(x_range,
                recur_ratios,
                'c.-')
        ax.set_xticklabels(x_names,
                           rotation=80)
        ax.set_xticks(np.arange(len(x_names)))
        ax.set_ylim([-0.1, 1.2])
        ax.title.set_text('Accumulative Feature Impact')
        fig.savefig('accum_feature_impact.png',
                    bbox_inches='tight')

    def house_price_histograme(self):
        price = self.y.copy().values
        plt.figure(figsize=(16, 9))
        plt.hist(x = price,
                 bins = 80,
                 facecolor = 'c',
                 alpha = 1)
        plt.xlabel('House Price')
        plt.ylabel('Count')
        plt.grid(True)
        plt.ylim([0, 140])
        plt.title('House Price Distribution')
        plt.savefig('price_distribution.png')