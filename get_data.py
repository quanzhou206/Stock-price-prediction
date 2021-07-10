import time
from get_factor_day import *
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import os


def get_data(code, pre_date, end_date, rate, window, factor_flag):
    stock_code = normalize_code(code)
    pre_num = 1  # 预测未来一个月数据
    path_name = 'stock_data_day_' + code + '.csv'
    folder = os.path.exists(path_name)
    if not folder:

        stock_data = get_price(stock_code, count=pre_date, end_date=end_date, frequency='daily',
                               fields=['open', 'close', 'high', 'low', 'volume', 'money'])
        if factor_flag:
            stock_data = get_factor(stock_code, stock_data)
        stock_data.to_csv(path_name, index=True)
        exit()
    else:
        stock_data = pd.read_csv(path_name)

    if factor_flag:
        stock_data.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'money', 'MACD', 'DIFF', 'DEA', 'WVAD',
                              'OBV', 'RSI', 'MA10', 'MA20', 'MA60', 'BOLL_UB', 'BOLL_MB', 'BOLL_LB']
    else:
        stock_data.columns = ['date', 'open', 'close', 'high', 'low', 'volume', 'money']

    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data = stock_data.set_index(['date'])
    stock_data['close_After_day'] = stock_data['close'].shift(-pre_num)
    stock_data = stock_data.dropna()
    X = stock_data.drop(['close_After_day'], axis=1)
    y = stock_data['close_After_day']

    training_data_len = np.int(len(X) * rate)

    X_train = X.iloc[:training_data_len]
    y_train = y.iloc[:training_data_len]
    X_test = X.iloc[training_data_len:]
    X_test_LSTM = X.iloc[training_data_len - window:]
    y_test = y.iloc[training_data_len:]
    return stock_data, X_train, y_train, X_test, y_test, X_test_LSTM


def data_normalize(X_train, y_train, X_test, y_test, window):
    sx = MinMaxScaler(feature_range=(0, 1))
    sy = MinMaxScaler(feature_range=(0, 1))

    # 数据归一化
    scaled_X_train = sx.fit_transform(X_train)
    scaled_X_test = sx.transform(X_test)
    scaled_y_train = sy.fit_transform(np.array(y_train).reshape(-1, 1))
    scaled_y_test = sy.transform(np.array(y_test).reshape(-1, 1))
    # 构建训练数据
    X = []
    y = []
    for i in range(window, len(X_train)):
        X.append(scaled_X_train[i - window+1:i+1, :])
        y.append(scaled_y_train[i])
    X, y = np.array(X), np.array(y)

    # LSTM expects the data to be 3 dimensional
    X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2]))

    Xt = []
    for i in range(window, len(X_test)):
        Xt.append(scaled_X_test[i - window+1:i+1, :])

    Xt, yt = np.array(Xt), np.array(scaled_y_test)
    Xt = np.reshape(Xt, (Xt.shape[0], Xt.shape[1], Xt.shape[2]))
    return X, y, Xt, yt, sx, sy

def get_Ridge_data(Ridge_data,LSTM,LSTM_CNN,CNN_LSTM,random,Adaboost):
    Ridge_data.loc[:, 'LSTM'] = LSTM
    Ridge_data.loc[:, 'LSTM_CNN'] = LSTM_CNN
    Ridge_data.loc[:, 'CNN_LSTM'] = CNN_LSTM
    Ridge_data.loc[:, 'random'] = random
    Ridge_data.loc[:, 'Adaboost'] = Adaboost
    return Ridge_data