# -*- coding: utf-8 -*-

import numpy as np
import statistics
from datasets import ReadData
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from relation_plot import RelationPlot
from collections import Counter


class FeaturePreprocess(ReadData):

    def __init__(self, trainFile, testFile, ispred=False):
        super(FeaturePreprocess, self).__init__(trainFile=trainFile,
                                                testFile=testFile)
        self.rownum = self.dataframe.shape[0]
        self.colnum = self.dataframe.shape[1]
        self.ispred = ispred

    def _missing_value(self):
        """
        Count number of missing values in each column.
        :return:
        Series
        """
        dataframe = self.dataframe.copy()
        missing_count = dataframe.isnull().sum()
        return missing_count

    def _remove_NA_col(self):
        """
        Remove columns with more than 40% missing values.
        !! This part is only for predicting data.
        :return:
        """
        missing_count = self._missing_value()
        col_remove = missing_count[missing_count>self.rownum * .4]
        colname_remove = col_remove.index.tolist()
        self.colname_remove_prep = col_remove
        df = self.dataframe.copy()
        df = df.drop(colname_remove, axis = 1)
        self.numeric_features = df.dtypes[df.dtypes != "object"].index
        self.dataframe = df
        return df

    def _basic_encoding(self):
        df = self._remove_NA_col()
        qual_cond_dict = {'NA': 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        df['BldgType'] = df['BldgType'].map({'1Fam': 5, '2fmCon': 4, 'Duplex': 3,
                                             'TwnhsE': 2, 'Twnhs': 1})
        df['HouseStyle'] = df['HouseStyle'].map({'SLvl': 6, 'SFoyer': 5,
                                                 '2.5Fin': 4, '2.5Unf': 3.5,
                                                 '2Story': 3, '1.5Fin': 2,
                                                 '1.5Unf': 1.5, '1Story': 1})
        df['ExterQual'] = df['ExterQual'].map(qual_cond_dict)
        df['ExterCond'] = df['ExterCond'].map(qual_cond_dict)
        df['BsmtQual'] = df['BsmtQual'].map(qual_cond_dict)
        df['BsmtCond'] = df['BsmtCond'].map(qual_cond_dict)

        df['BsmtExposure'] = df['BsmtExposure'].map({'NA': 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4})
        df['BsmtFinType1'] = df['BsmtFinType1'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4,
                                                     'Rec': 3, 'LwQ': 2, 'Unf': 1,
                                                     'NA': 0})
        df['BsmtFinType2'] = df['BsmtFinType2'].map({'GLQ': 6, 'ALQ': 5, 'BLQ': 4,
                                                     'Rec': 3, 'LwQ': 2, 'Unf': 1,
                                                     'NA': 0})
        df['LotShape'] = df['LotShape'].map({'Reg': 4, 'IR1': 3,
                                             'IR2': 2, 'IR3': 1})
        df['GarageType'] = df['GarageType'].map({'2Types': 6, 'Attchd': 5,
                                                 'Basment': 4, 'BuiltIn': 3,
                                                 'CarPort': 2, 'Detchd': 1,
                                                 'NA': 0})
        df['PavedDrive'] = df['PavedDrive'].map({'Y': 1, 'P': 0.5, 'N': 0})
        df['GarageFinish'] = df['GarageFinish'].map({'Fin': 3, 'RFn': 2,
                                                     'Unf': 1, 'NA': 0})
        df['GarageQual'] = df['GarageQual'].map(qual_cond_dict)
        df['GarageCond'] = df['GarageCond'].map(qual_cond_dict)
        df['Foundation'] = df['Foundation'].map({'PConc': 4, 'CBlock': 3, 'BrkTil': 2,
                                                 'Slab': 1, 'Stone': 1, 'Wood': 1})

        self.dataframe = df
        return df


    def _fill_NA(self):
        df = self._basic_encoding()
        df['MSZoning'].fillna(statistics.mode(df['MSZoning']),
                              inplace=True)

        LotFront_NA_index = df['LotFrontage'].isnull()
        LR = LinearRegression()
        LR.fit(df['LotArea'][~LotFront_NA_index].values.reshape(-1,1),
               df['LotFrontage'][~LotFront_NA_index])
        LotFront_pred = LR.predict(df['LotArea'][LotFront_NA_index].values.reshape(-1,1))
        df.ix[LotFront_NA_index, 'LotFrontage'] = list(map(int, np.round(LotFront_pred)))
        del LR

        df['Utilities'].fillna(statistics.mode(df['Utilities']),
                               inplace=True)

        df['Exterior1st'].fillna(statistics.mode(df['Exterior1st']),
                                 inplace=True)
        df['Exterior2nd'].fillna(statistics.mode(df['Exterior2nd']),
                                 inplace=True)

        df['MasVnrArea'].fillna(0,
                                inplace=True)
        df['MasVnrType'].fillna('None',
                                inplace=True)
        df.ix[:, ['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']] = \
            df[['BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF']].fillna(0)
        df.ix[:, ['BsmtFullBath', 'BsmtHalfBath']] = \
            df[['BsmtFullBath', 'BsmtHalfBath']].fillna(0)
        df['Electrical'].fillna(statistics.mode(df['Electrical']),
                                inplace=True)
        df['KitchenQual'].fillna(statistics.mode(df['KitchenQual']),
                                 inplace=True)
        df['Functional'].fillna(statistics.mode(df['Functional']),
                                inplace=True)
        df['SaleType'].fillna(statistics.mode(df['SaleType']),
                              inplace=True)

        BsmtQual_NA_index = df['BsmtQual'].isnull()
        LR = LinearRegression()
        LR.fit(df.ix[~BsmtQual_NA_index, ['ExterQual', 'OverallQual']],
               df.ix[~BsmtQual_NA_index, 'BsmtQual'])
        BsmtQual_pred = LR.predict(df.ix[BsmtQual_NA_index,
                                         ['ExterQual', 'OverallQual']])
        df.ix[BsmtQual_NA_index, 'BsmtQual'] = list(map(int, np.round(BsmtQual_pred)))
        del LR

        df['BsmtCond'].fillna(statistics.mode(df['BsmtCond']),
                              inplace=True)

        BsmtFinType1_NA_index = df['BsmtFinType1'].isnull()
        LR = LinearRegression()
        LR.fit(df.ix[~BsmtFinType1_NA_index, 'BsmtFinSF1'].values.reshape(-1,1),
               df.ix[~BsmtFinType1_NA_index, 'BsmtFinType1'])
        BsmtFinType1_pred = LR.predict(df.ix[BsmtFinType1_NA_index,
                                             'BsmtFinSF1'].values.reshape(-1,1))
        df.ix[BsmtFinType1_NA_index, 'BsmtFinType1'] = list(map(int, np.round(BsmtFinType1_pred)))
        del LR

        df['BsmtFinType2'].fillna(statistics.mode(df['BsmtFinType2']),
                                  inplace=True)
        df['BsmtExposure'].fillna(statistics.mode(df['BsmtExposure']),
                                  inplace=True)


        df['GarageCars'].fillna(statistics.mode(df['GarageCars']),
                                inplace=True)
        df['GarageArea'].fillna(statistics.mode(df['GarageArea']),
                                inplace=True)
        df['GarageType'].fillna(statistics.mode(df['GarageType']),
                                inplace=True)
        GarageYrBlt_NA_index = df['GarageYrBlt'].isnull()
        df.ix[GarageYrBlt_NA_index, 'GarageYrBlt'] = df.ix[GarageYrBlt_NA_index, 'YearBuilt']
        df['GarageQual'].fillna(statistics.mode(df['GarageQual']),
                                inplace=True)
        df['GarageCond'].fillna(statistics.mode(df['GarageCond']),
                                inplace=True)
        df['GarageFinish'].fillna(statistics.mode(df['GarageFinish']),
                                  inplace=True)

        # print(df.isnull().sum()[df.isnull().sum() > 0])
        #
        # RelationPlot(data=df,
        #              x='YearRemodAdd', y='GarageFinish',
        #              is_discrete=True,
        #              output_dir='feature_relation')
        self.dataframe = df
        df = _encoding(df, ['MSZoning', 'Street', 'LandContour', 'Utilities', 'LotConfig', 'LandSlope',
                            'Neighborhood', 'Condition1', 'Condition2', 'RoofStyle', 'RoofMatl',
                            'Exterior1st', 'Exterior2nd', 'MasVnrType', 'Heating', 'HeatingQC', 'CentralAir',
                            'Electrical', 'KitchenQual', 'Functional', 'SaleType', 'SaleCondition'])
        return df


    def _remove_outliers(self):
        self._remove_NA_col()
        df = self.dataframe
        df.drop(df[df["GrLivArea"] > 4000].index,
                inplace = True)
        df.drop(df[df['LotFrontage'] > 160].index,
                inplace = True)
        df.drop(df[df['LotArea'] > 60000].index,
                inplace=True)
        df.drop(df[df['MasVnrArea'] > 1250].index,
                inplace=True)
        df.drop(df[df['TotalBsmtSF'] > 2800].index,
                inplace=True)
        df.drop(df[df['1stFlrSF'] > 2500].index,
                inplace=True)
        df.drop(df[df['GarageArea'] > 1200].index,
                inplace=True)
        df.drop(df[df['WoodDeckSF'] > 600].index,
                inplace=True)
        df.drop(df[df['OpenPorchSF'] > 350].index,
                inplace=True)

        self.dataframe = df


    # def _convert_(self):
    #     """
    #     Convert string to int in each column.
    #     :return:
    #     """
    #     self._domain_knwl_encod()
    #     dataframe = self.dataframe.copy()
    #     colnames = dataframe.columns
    #
    #     for name in colnames:
    #         column = dataframe[name]
    #         dataframe[name] = _convert_column(column=column)
    #
    #     self.dataframe = dataframe


    def _domain_knwl_encod(self):
        df = self._fill_NA()
        df["RegularLotShape"] = (df['LotShape'] == 4) * 1
        df["IsLandLevel"] = (df['LandContour'] == "Lvl") * 1
        df['AllUtilities'] = (df['Utilities'] == 'AllPub') * 1
        df['GentleLandSlope'] = (df['LandSlope'] == 3) * 1
        df["IsElectricalSBrkr"] = (df['Electrical'] == "SBrkr") * 1
        df["Has2ndFloor"] = (df['2ndFlrSF'] == 0) * 1
        df["IsLowQual"] = (df['LowQualFinSF'] == 0) * 1
        df["IsGarageDetached"] = (df['GarageType'] == 1) * 1
        df["VeryNewHouse"] = (df['YearBuilt'] == df["YrSold"]) * 1
        df['BuiltAge'] = 2011 - df['YearBuilt']
        df['RemodelAge'] = 2011 - df['YearRemodAdd']
        # df.drop(['YearBuilt', 'YearRemodAdd', 'LotShape', 'LandContour',
        #                 'Utilities', 'LandSlope', 'Electrical', 'GarageType'],
        #                axis = 1,inplace = True)
        df["HighSeason"] = df["MoSold"].replace(
            {1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 1, 8: 0, 9: 0, 10: 0, 11: 0, 12: 0})
        df["MSSubClass"] = df["MSSubClass"].replace(
            {20: 1, 30: 0, 40: 0, 45: 0, 50: 0, 60: 1, 70: 0, 75: 0, 80: 0, 85: 0,
             90: 0, 120: 1, 150: 0, 160: 0, 180: 0, 190: 0})
        df.loc[df.Neighborhood == 'NridgHt', "Neighborhood_Good"] = 1
        df.loc[df.Neighborhood == 'Crawfor', "Neighborhood_Good"] = 1
        df.loc[df.Neighborhood == 'StoneBr', "Neighborhood_Good"] = 1
        df.loc[df.Neighborhood == 'Somerst', "Neighborhood_Good"] = 1
        df.loc[df.Neighborhood == 'NoRidge', "Neighborhood_Good"] = 1
        df["Neighborhood_Good"].fillna(0, inplace=True)
        # df.drop(['MoSold'], axis=1, inplace=True)

        qual_dict = {'NA': 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
        df['LotShape'] = df['LotShape'].map({'Reg': 4, 'IR1': 3,
                                             'IR2': 2, 'IR3': 1})
        df['LandSlope'] = df['LandSlope'].map({'Gtl': 3, 'Mod':2, 'Sev': 1})
        df['BldgType'] = df['BldgType'].map({'1Fam':5, '2FmCon':4, 'Duplx':3,
                                             'TwnhsE': 2, 'TwnhsI':1})
        df['HouseStyle'] = df['HouseStyle'].map({'SLvl':6, 'SFoyer':5,
                                                 '2.5Fin':4, '2.5Unf': 3.5,
                                                 '2Story':3, '1.5Fin': 2,
                                                 '1.5Unf': 1.5, '1Story': 1})

        df['BsmtFinType1'] = df['BsmtFinType1'].map({'GLQ':6, 'ALQ':5, 'BLQ':4,
                                                     'Rec':3, 'LwQ':2, 'Unf':1,
                                                     'NA':0})
        df['BsmtFinType2'] = df['BsmtFinType2'].map({'GLQ':6, 'ALQ':5, 'BLQ':4,
                                                     'Rec':3, 'LwQ':2, 'Unf':1,
                                                     'NA':0})
        df['HeatingQC'] = df['HeatingQC'].map(qual_dict)
        df['CentralAir'] = df['CentralAir'].map({'Y':1, 'N':0})
        df['KitchenQual'] = df['KitchenQual'].map(qual_dict)
        df['GarageType'] = df['GarageType'].map({'2Types':6, 'Attchd':5,
                                                 'Basment':4, 'BuiltIn':3,
                                                 'CarPort':2, 'Detchd':1,
                                                 'NA':0})
        df['GarageFinish'] = df['GarageFinish'].map({'Fin':3, 'RFn':2,
                                                     'Unf':1, 'NA':0})
        df['GarageQual'] = df['GarageQual'].map(qual_dict)
        df['GarageCond'] = df['GarageCond'].map(qual_dict)
        df["Functional"] = df["Functional"].map({None: 0, "Sal": 1, "Sev": 2,
                                                 "Maj2": 3, "Maj1": 4, "Mod": 5,
                                                 "Min2": 6, "Min1": 7, "Typ": 8})
        df['SaleCondition_PriceDown'] = df['SaleCondition'].map({'Abnorml': 1, 'Alloca': 1,
                                                                 'AdjLand': 1, 'Family': 1,
                                                                 'Normal': 0, 'Partial': 0})
        df["BoughtOffPlan"] = df['SaleCondition'].map({"Abnorml": 0, "Alloca": 0, "AdjLand": 0,
                                                       "Family": 0, "Normal": 0, "Partial": 1})

        df["SimplOverallQual"] = df['OverallQual'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2,
                                                        6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
        df["SimplOverallCond"] = df['OverallCond'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2,
                                                        6: 2, 7: 3, 8: 3, 9: 3, 10: 3})
        df["SimplGarageCond"] = df['GarageCond'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        df["SimplGarageQual"] = df['GarageQual'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        df["SimplFunctional"] = df['Functional'].map({1: 1, 2: 1, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 4})
        df["SimplKitchenQual"] = df['KitchenQual'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        df["SimplHeatingQC"] = df['HeatingQC'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        df["SimplBsmtFinType1"] = df['BsmtFinType1'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
        df["SimplBsmtFinType2"] = df['BsmtFinType2'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2, 6: 2})
        df["SimplBsmtCond"] = df['BsmtCond'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        df["SimplBsmtQual"] = df['BsmtQual'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        df["SimplExterCond"] = df['ExterCond'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})
        df["SimplExterQual"] = df['ExterQual'].map({1: 1, 2: 1, 3: 1, 4: 2, 5: 2})

        self.dataframe = df

    def run(self):
        pass




def _encoding(dataframe, column_name):
    """
    Convert string to int in one column.
    :param dataframe: pandas.DataFrame
           column_name: string or list
                Column name(s) in dataframe.
    :return: Encoded dataframe.
    """
    le = LabelEncoder()
    le_transform = le.fit_transform
    try:
        dataframe[column_name] = dataframe[column_name].apply(le_transform)
    except Exception as e:
        print('Encoding failure for following reasons:', e)
    return dataframe




if __name__ == '__main__':
    FeaturePreprocess(trainFile='train.csv',
                      testFile='test.csv')._fill_NA()