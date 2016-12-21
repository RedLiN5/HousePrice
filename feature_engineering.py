# -*- coding: utf-8 -*-

from datasets import ReadData

class FeatureEngin(ReadData):

    def __init__(self):
        super(FeatureEngin, self).__init__()
        self.run()
        self.rownum = self.dataframe.shape[0]
        self.colnum = self.dataframe.shape[1]

    def _missing_value(self):
        """
        Count number of missing values in each column.
        :return:
        Series
        """
        dataframe = self.dataframe
        missing_count = dataframe.isnull().sum()
        return missing_count

    def _remove_missing(self):
        """
        Remove columns with more than 40% missing values.
        :return:
        """
        missing_count = self._missing_value()
        col_remove = missing_count[missing_count>self.colnum * .4]
        colname_remove = col_remove.index.tolist()
        self.dataframe = self.dataframe.drop(colname_remove,
                                             axis = 1)

    def _convert_column(self, column):
        """
        Convert string to int in one column.
        :param column: One column in dataframe.
        :return: Converted column.
        """
        column_values = column[~column.isnull()]
        is_string = column_values.apply(lambda x: isinstance(x, str)).any()
        if is_string:
            unique_values = list(set(column_values))
            int_length = len(unique_values)
            for i in range(int_length):
                value = unique_values[i]
                column[column == value] = i+1

        return column

    def _convert(self):
        """
        Convert string to int in each column.
        :return:
        """
        self._remove_missing()
        dataframe = self.dataframe.copy()
        colnames = self.dataframe.columns
        value_count = [] * self.dataframe.shape[0]
        for name in colnames:
            column = dataframe[name]
            dataframe[name] = self._convert_column(column=column)

        self.dataframe = dataframe
