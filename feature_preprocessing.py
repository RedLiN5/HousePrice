# -*- coding: utf-8 -*-

import statistics
from datasets import ReadData

class FeaturePreprocess(ReadData):

    def __init__(self, filename, istest=False):
        super(FeaturePreprocess, self).__init__(filname=filename)
        self.load()
        self.rownum = self.dataframe.shape[0]
        self.colnum = self.dataframe.shape[1]
        self.istest = istest

    def _missing_value(self):
        """
        Count number of missing values in each column.
        :return:
        Series
        """
        dataframe = self.dataframe.copy()
        missing_count = dataframe.isnull().sum()
        return missing_count

    def _remove_missing_(self):
        """
        Remove columns with more than 40% missing values.
        :return:
        """
        missing_count = self._missing_value()
        col_remove = missing_count[missing_count>self.rownum * .4]
        colname_remove = col_remove.index.tolist()
        self.colname_remove_prep = col_remove
        self.dataframe = self.dataframe.drop(colname_remove,
                                             axis = 1)

    def _convert_column(self, column):
        """
        Convert string to int in one column.
        :param column: pandas.Series
                       One column in dataframe.
        :return: Converted column.
        """
        column_values = column[~column.isnull()]
        is_string = column_values.apply(lambda x: isinstance(x, str)).any()

        if is_string:
            unique_values = list(set(column_values))
            int_length = len(unique_values)

            for i in range(int_length):
                value = unique_values[i]
                column.loc[column == value] = i+1

        return column

    def _convert_(self):
        """
        Convert string to int in each column.
        :return:
        """
        if ~self.istest:
            self._remove_missing_()
        dataframe = self.dataframe.copy()
        colnames = self.dataframe.columns
        value_count = [] * self.dataframe.shape[0]

        for name in colnames:
            column = dataframe[name]
            dataframe[name] = self._convert_column(column=column)

        self.dataframe = dataframe

    def _interpolate_(self):
        """
        Interpolate missing values with mode.
        :return:
        """
        self._convert_()
        dataframe = self.dataframe.copy()
        missing_count = dataframe.isnull().sum()
        col_missing = missing_count[missing_count > 0]
        colnames_missing = col_missing.index.tolist()

        for colname in colnames_missing:
            column = dataframe[colname]
            index_missing = column.isnull()
            mode = statistics.mode(column[~index_missing])
            column = column.fillna(mode)
            dataframe[colname] = column

        self.dataframe = dataframe

    def run_preprocessor(self):
        self._interpolate_()