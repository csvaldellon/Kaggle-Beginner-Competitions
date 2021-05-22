import pandas as pd, matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score


def find_x(path, is_df=False, quant_only=False, categ_only=False):
    data = pd.read_csv(path)
    cols_ok = ['MSSubClass',  'Street', 'Alley', 'LotShape',
                          'LotConfig', 'LandSlope',
                          'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                          'RoofMatl',
                          'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual',
                          'BsmtExposure',
                          'BsmtFinType1',
                          'Heating', 'HeatingQC', 'CentralAir',
                          'FireplaceQu',
                          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
                          'Fence', 'MiscFeature', 'SaleCondition']
    cols_ok = ['ExterQual', 'Foundation', 'BsmtQual', 'HeatingQC', 'FireplaceQu', 'GarageType',
               'GarageFinish']  # 0.09
    # cols_ok = ['ExterQual', 'Foundation', 'BsmtQual', 'BsmtExposure', 'HeatingQC', 'FireplaceQu', 'GarageType',
               # 'GarageFinish']  # 0.1
    # cols_ok = ['ExterQual', 'BsmtQual', 'FireplaceQu', 'GarageType', 'GarageFinish']  # 0.2
    # cols_ok = ['ExterQual', 'BsmtQual']  # 0.3
    # cols_ok = ['ExterQual']  # 0.4

    x_categorical_ok = data[cols_ok]
    # print(x_categorical_ok.isna().sum())

    cols_na = ["MSZoning", "LandContour", "Utilities","Exterior1st", "Exterior2nd", "MasVnrType",
                              "Electrical", "KitchenQual", "Functional"]
    cols_na = ["KitchenQual"]  # 0.2, 0.3, 0.1, 0.09
    # cols_na = []  # 0.4

    x_categorical_na = data[cols_na]
    # no "BsmtCond","BsmtFinType2" yet
    # print(x_categorical_na.isna().sum())
    x_categorical_fill = x_categorical_na.apply(lambda z: z.fillna(z.value_counts().index[0]))
    # print(x_categorical_fill.isna().sum())
    x_categorical = x_categorical_ok.join(x_categorical_fill)
    # print(x_categorical.isna().sum())
    x_categorical = x_categorical.values
    Le = LabelEncoder()
    for i in range(len(x_categorical[0])):
        x_categorical[:, i] = Le.fit_transform(x_categorical[:, i])
    x_categorical = pd.DataFrame(x_categorical)
    x_categorical.columns = cols_ok + cols_na
    # print(x_categorical.isna().sum())
    if categ_only:
        return x_categorical
    x_quantitative_ok = data[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
                              '1stFlrSF', '2ndFlrSF',
                           'LowQualFinSF',
                           'GrLivArea',
                              'FullBath', 'HalfBath', 'BedroomAbvGr',
                           'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces',
                              'WoodDeckSF',
                           'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold',
                           'YrSold']]
    x_quantitative_ok = data[['OverallQual', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'GrLivArea',
                              'FullBath', 'HalfBath', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF',
                              'OpenPorchSF']]  # 0.09
    # x_quantitative_ok = data[['OverallQual', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'GrLivArea', 'FullBath',
                              # 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF']]  # 0.1
    # x_quantitative_ok = data[['OverallQual', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', 'GrLivArea', 'FullBath',
                              # 'TotRmsAbvGrd', 'Fireplaces']]  # 0.2
    # x_quantitative_ok = data[['OverallQual', '1stFlrSF', 'GrLivArea', 'FullBath',  'TotRmsAbvGrd']]  # 0.3
    # x_quantitative_ok = data[['OverallQual', 'GrLivArea']]  # 0.4

    # print(x_quantitative_ok.isna().sum())
    x_quantitative_na = data[["LotFrontage", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", "BsmtUnfSF", "TotalBsmtSF",
                              "BsmtFullBath", "BsmtHalfBath", "GarageYrBlt", "GarageCars", "GarageArea"
                              ]]
    x_quantitative_na = data[['MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GarageYrBlt', 'GarageCars',
                              'GarageArea']]  # 0.09
    # x_quantitative_na = data[['LotFrontage', 'MasVnrArea', 'BsmtFinSF1', 'TotalBsmtSF', 'GarageYrBlt', 'GarageCars',
                              # 'GarageArea']]  # 0.1
    # x_quantitative_na = data[["MasVnrArea", "TotalBsmtSF", "GarageYrBlt", "GarageCars", "GarageArea"]]  # 0.2
    # x_quantitative_na = data[["TotalBsmtSF", "GarageCars", "GarageArea"]]  # 0.3, 0.4

    # print(x_quantitative_na.isna().sum())
    x_quantitative_fill = x_quantitative_na.fillna(x_quantitative_na.mean())
    # print(x_quantitative_fill.isna().sum())
    x_quantitative = x_quantitative_ok.join(x_quantitative_fill)
    if quant_only:
        return x_quantitative
    # print(x_quantitative.isna().sum())
    x = x_categorical.join(x_quantitative)
    if is_df:
        return x
    return x.values


def find_submit(train, test, submit):
    x_train = find_x(train)
    y_train = pd.read_csv(train)[['SalePrice']].values
    x_test = find_x(test)

    l_reg = linear_model.LinearRegression()
    model = l_reg.fit(x_train, y_train)
    prediction = model.predict(x_test)
    prediction = pd.DataFrame(prediction)
    prediction.columns = ['SalePrice']
    prediction.to_csv(submit)


def find_accuracy(train):
    x = find_x(train)
    y = pd.read_csv(train)[['SalePrice']].values
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    l_reg = linear_model.LinearRegression()
    l_reg.fit(x_train, y_train)
    prediction = l_reg.predict(x_test)
    accuracy_1 = r2_score(y_test, prediction)

    print('accuracy using l_reg: ', accuracy_1)

# find_submit(train, test, submit)
# find_accuracy(train)


def column_finder():
    features = find_x(train, is_df=True)
    column_list = []
    for column in features.keys():
        x = features[[column]].values
        y = pd.read_csv(train)[['SalePrice']].values
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        l_reg = linear_model.LinearRegression()
        l_reg.fit(x_train, y_train)
        prediction = l_reg.predict(x_test)
        accuracy_1 = r2_score(y_test, prediction)
        if accuracy_1 >= 0.0:
            column_list += [column]
            print('accuracy using l_reg: ', accuracy_1, " ", str(column))
    print(column_list)


train = 'C:/Users/Val/Desktop/House Prices/train.csv'
test = 'C:/Users/Val/Desktop/House Prices/test.csv'
submit = 'C:/Users/Val/Desktop/House Prices/submission l_reg 8.csv'

find_accuracy(train)
find_submit(train, test, submit)
# column_finder()
