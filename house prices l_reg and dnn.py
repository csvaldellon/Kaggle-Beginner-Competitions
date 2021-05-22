import pandas as pd, numpy as np, tensorflow as tf
from sklearn import linear_model, neural_network
from sklearn.preprocessing import LabelEncoder, scale
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from tensorflow.keras import layers, optimizers, Sequential, callbacks


def find_x(path):
    data = pd.read_csv(path)
    x_categorical = data[['MSSubClass', 'Street', 'Alley', 'LotShape', 'LotConfig', 'LandSlope',
                          'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                          'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtExposure',
                          'BsmtFinType1', 'Heating', 'HeatingQC', 'CentralAir', 'FireplaceQu',
                          'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
                          'Fence', 'MiscFeature', 'SaleCondition']]
    # print(x_categorical.isna().sum())
    x_categorical = x_categorical.values
    Le = LabelEncoder()
    for i in range(len(x_categorical[0])):
        x_categorical[:, i] = Le.fit_transform(x_categorical[:, i])
    x_categorical = pd.DataFrame(x_categorical)
    x_categorical.columns = ['MSSubClass', 'Street', 'Alley', 'LotShape', 'LotConfig', 'LandSlope',
                             'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 'RoofStyle',
                             'RoofMatl', 'ExterQual', 'ExterCond', 'Foundation', 'BsmtQual', 'BsmtExposure',
                             'BsmtFinType1', 'Heating', 'HeatingQC', 'CentralAir', 'FireplaceQu',
                             'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'PavedDrive', 'PoolQC',
                             'Fence', 'MiscFeature', 'SaleCondition']
    # print(x_categorical.isna().sum())
    x_quantitative = data[['LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd', '1stFlrSF', '2ndFlrSF', 'LowQualFinSF',
                           'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr',
                           'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 'WoodDeckSF',
                           'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold',
                           'YrSold']]
    # print(x_categorical.isna().sum())
    x = x_categorical.join(x_quantitative).values
    return x


train = 'D:/OTHERS/Desktop/Others/Sem Break 2020-2021 Programming Projects/Project 2 - Dec 30 to Jan 6 - Machine Learning/Kaggle/House Prices/train.csv'
test = 'D:/OTHERS/Desktop/Others/Sem Break 2020-2021 Programming Projects/Project 2 - Dec 30 to Jan 6 - Machine Learning/Kaggle/House Prices/test.csv'
x_train = find_x(train)
# x_train = scale(x_train)
y_train = pd.read_csv(train)[['SalePrice']].values
x_test = find_x(test)

# ar = x_test
# ar_nan = np.where(np.isnan(ar))
# print(ar_nan)

l_reg = linear_model.LinearRegression()
# model = l_reg.fit(x_train, y_train)
# prediction = model.predict(x_test)
# prediction = pd.DataFrame(prediction)
# prediction.to_csv('C:/Users/Val/Desktop/House Prices/submission l_reg.csv')

x_train = scale(x_train)
x_test = scale(x_test)
dnn = neural_network.MLPRegressor(solver='adam', activation='relu', hidden_layer_sizes=(64, 64),
                                  max_iter=12000, verbose=True)
# early_stopping=True, verbose=True, tol=0.000001, n_iter_no_change=1000)
# dnn.fit(x_train, np.ravel(y_train))
# prediction = dnn.predict(x_test)
# prediction = pd.DataFrame(prediction)
# prediction.to_csv('C:/Users/Val/Desktop/House Prices/submission sklearn dnn.csv')

x = find_x(train)
y = pd.read_csv(train)[['SalePrice']].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# l_reg = linear_model.LinearRegression()
# l_reg.fit(x_train, y_train)
# prediction = model.predict(x_test)
# accuracy_1 = r2_score(y_test, prediction)

x_train = scale(x_train)
x_test = scale(x_test)

dnn = neural_network.MLPRegressor(solver='adam', activation='relu', hidden_layer_sizes=(64, 64),
                                  max_iter=12000, verbose=True)
# early_stopping=True, verbose=True, tol=0.000001, n_iter_no_change=1000)
# dnn.fit(x_train, np.ravel(y_train))
# prediction = dnn.predict(x_test)
# accuracy_2 = r2_score(y_test, prediction)


model = Sequential([
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1)
])
print(model.history)
model.compile(loss='mse', optimizer=optimizers.Adam(0.001))
model.fit(tf.convert_to_tensor(x_train), tf.convert_to_tensor(y_train), epochs=10000, validation_split=0.2,
          callbacks=[callbacks.EarlyStopping(patience=10, monitor='val_loss')])
prediction = model.predict(x_test)
accuracy_3 = r2_score(y_test, prediction)

# print('accuracy using l_reg: ', accuracy_1)        # 0.8394
# print('accuracy using sklearn dnn: ', accuracy_2)  # 0.7717
print('accuracy using keras dnn: ', accuracy_3)    # 0.7939
