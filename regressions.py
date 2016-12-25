# -*- coding: utf-8 -*-

from feature_engineering import FeatureEngin


class Regressions(FeatureEngin):

    def __init__(self):
        super(Regressions, self).__init__()
        self.feature_names = FeatureEngin.start(self)

    def run(self):
        # TODO(Leslie) add last function to run
        pass
