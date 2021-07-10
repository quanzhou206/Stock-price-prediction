import tqdm
from get_factor_day import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from get_data import *
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.linear_model import BayesianRidge
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import *
import pandas as pd
import os


def random_forest(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(n_estimators=500, min_samples_split=5)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    train_pre = model.predict(X_train)
    print(f'RandomForestRegressor:')
    print(f'R2: {r2_score(y_test, predictions):.2f}')
    print(f'MAE: {mean_absolute_error(y_test, predictions):.2f}')
    print(f'MSE: {mean_squared_error(y_test, predictions):.2f}')
    print('\n')
    return predictions, train_pre


def Adaboost(X_train, y_train, X_test, y_test):
    model = AdaBoostRegressor(n_estimators=500)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    train_pre = model.predict(X_train)
    print(f'AdaBoostRegressor:')
    print(f'R2: {r2_score(y_test, predictions):.2f}')
    print(f'MAE: {mean_absolute_error(y_test, predictions):.2f}')
    print(f'MSE: {mean_squared_error(y_test, predictions):.2f}')
    print('\n')
    return predictions, train_pre


def LSTM_pre(X_train, y_train, X_test, y_test, window):
    # 对数据进行归一化
    X_train, y_train, X_test, y_test, sx, sy = data_normalize(X_train, y_train, X_test, y_test, window)
    # 构建模型
    model = Sequential()
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam',
                  loss='mean_squared_error')

    if os.path.exists('LSTM_model.h5'):
        model.fit(X_train, y_train, epochs=0, )
        model.load_weights('LSTM_model.h5')
    else:
        history = model.fit(X_train, y_train,
                            batch_size=32,
                            epochs=100,
                            validation_data=(X_test, y_test),
                            shuffle=True,
                            )
        model.save_weights('LSTM_model.h5')

        # 绘制训练日志
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title('LSTM训练日志', fontsize='12')
        plt.ylabel('loss', fontsize='10')
        plt.xlabel('epoch', fontsize='10')
        plt.legend()
        plt.show()

    predictions = model.predict(X_test)
    predictions = sy.inverse_transform(predictions)
    train_pre = model.predict(X_train)
    train_pre = sy.inverse_transform(train_pre)
    print(f'\nLSTM:')
    print(f'R2: {r2_score(y_test, predictions):.2f}')
    print(f'MAE: {mean_absolute_error(y_test, predictions):.2f}')
    print(f'MSE: {mean_squared_error(y_test, predictions):.2f}')
    print('\n')
    return predictions, train_pre


def LSTM_CNN_pre(X_train, y_train, X_test, y_test, window):
    # 对数据进行归一化
    X_train, y_train, X_test, y_test, sx, sy = data_normalize(X_train, y_train, X_test, y_test, window)
    # 构建模型
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Reshape((30, 32, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add((Dense(32)))
    model.add(Dense(1))
    model.compile(optimizer='adam',
                  loss='mean_squared_error')

    if os.path.exists('LSTM_CNN_model.h5'):
        model.load_weights('LSTM_CNN_model.h5')
    else:
        history = model.fit(X_train, y_train,
                            batch_size=32,
                            epochs=100,
                            validation_data=(X_test, y_test),
                            shuffle=True,
                            )
        model.save_weights('LSTM_CNN_model.h5')

        # 绘制训练日志
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title('LSTM训练日志', fontsize='12')
        plt.ylabel('loss', fontsize='10')
        plt.xlabel('epoch', fontsize='10')
        plt.legend()
        plt.show()

    predictions = model.predict(X_test)
    predictions = sy.inverse_transform(predictions)
    train_pre = model.predict(X_train)
    train_pre = sy.inverse_transform(train_pre)
    print(f'\nLSTM_CNN:')
    print(f'R2: {r2_score(y_test, predictions):.2f}')
    print(f'MAE: {mean_absolute_error(y_test, predictions):.2f}')
    print(f'MSE: {mean_squared_error(y_test, predictions):.2f}')
    print('\n')
    return predictions, train_pre


def CNN_LSTM_pre(X_train, y_train, X_test, y_test, window):
    # 对数据进行归一化
    X_train, y_train, X_test, y_test, sx, sy = data_normalize(X_train, y_train, X_test, y_test, window)
    # 构建模型
    model = Sequential()
    model.add(Reshape((X_train.shape[1], X_train.shape[2], 1)))
    model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))
    # layer=model.get_layer(index=2)
    # print(layer.output_shape)
    model.add(Flatten())
    model.add(Reshape((-1, 1)))
    model.add(LSTM(64))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(optimizer='adam',
                  loss='mean_squared_error')

    if os.path.exists('CNN_LSTM_model.h5'):
        history = model.fit(X_train, y_train, epochs=0)
        model.load_weights('CNN_LSTM_model.h5')
    else:
        history = model.fit(X_train, y_train,
                            batch_size=32,
                            epochs=100,
                            validation_data=(X_test, y_test),
                            shuffle=True,
                            )
        model.save_weights('CNN_LSTM_model.h5')

        # 绘制训练日志
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='test')
        plt.title('LSTM训练日志', fontsize='12')
        plt.ylabel('loss', fontsize='10')
        plt.xlabel('epoch', fontsize='10')
        plt.legend()
        plt.show()

    predictions = model.predict(X_test)
    predictions = sy.inverse_transform(predictions)
    train_pre = model.predict(X_train)
    train_pre = sy.inverse_transform(train_pre)
    print(f'\nCNN_LSTM:')
    print(f'R2: {r2_score(y_test, predictions):.2f}')
    print(f'MAE: {mean_absolute_error(y_test, predictions):.2f}')
    print(f'MSE: {mean_squared_error(y_test, predictions):.2f}')
    print('\n')
    return predictions, train_pre


def Ridge_pre(X_train, y_train, X_test, y_test):
    model = BayesianRidge()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    print(f'BayesianRidge:')
    print(f'R2: {r2_score(y_test, predictions):.2f}')
    print(f'MAE: {mean_absolute_error(y_test, predictions):.2f}')
    print(f'MSE: {mean_squared_error(y_test, predictions):.2f}')
    print('\n')
    return predictions


def LSTM_pre_init(X_train, y_train, X_test, y_test, window):
    # 对数据进行归一化
    X_train, y_train, X_test, y_test, sx, sy = data_normalize(X_train, y_train, X_test, y_test, window)
    # 构建模型
    model = Sequential()
    model.add(LSTM(32, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
    model.add(Reshape((30, 32, 1)))
    model.add(BatchNormalization())
    model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='relu', padding='same'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add((Dense(32)))
    model.add(Dense(1))
    model.compile(optimizer='adam',
                  loss='mean_squared_error')

    history = model.fit(X_train, y_train,
                        batch_size=32,
                        epochs=100,
                        validation_data=(X_test, y_test),
                        shuffle=True,
                        )

    # 绘制训练日志
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='test')
    plt.title('LSTM训练日志', fontsize='12')
    plt.ylabel('loss', fontsize='10')
    plt.xlabel('epoch', fontsize='10')
    plt.legend()
    plt.show()

    predictions = model.predict(X_test)
    predictions = sy.inverse_transform(predictions)
    print(f'\nLSTM:')
    print(f'R2: {r2_score(y_test, predictions):.2f}')
    print(f'MAE: {mean_absolute_error(y_test, predictions):.2f}')
    print(f'MSE: {mean_squared_error(y_test, predictions):.2f}')
    print('\n')
    return predictions


class Adaboost_integrate:
    def __init__(self, n_estimators=5):
        self.model_num = n_estimators
        self.emrate = []
        self.alpha = []
        self.model = []

    def Ada_fit(self, X_train, y_train, X_test, y_test, window):
        # 初始化权重
        X_train, y_train, X_test, y_test, sx, sy = data_normalize(X_train, y_train, X_test, y_test, window)
        self.weights = np.array([1.0 / X_train.shape[0]] * X_train.shape[0])
        model = Sequential()
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam',
                      loss='mean_squared_error')
        # 依次构建回归器模型
        for i in tqdm(range(self.model_num)):
            model_name = 'Ada_LSTM/LSTM_model_Adaboost_' + str(i) + '.h5'
            if os.path.exists(model_name):
                model.fit(X_train, y_train, epochs=0, )
                model.load_weights(model_name)
            else:
                model.fit(X_train, y_train,
                          batch_size=32,
                          epochs=100,
                          shuffle=True,
                          verbose=0,
                          sample_weight=self.weights
                          )
                model.save_weights(model_name)
            self.model.append(model_name)
            # 获取该模型在训练集上预测值
            predictions = model.predict(X_train)
            # 计算该模型最大误差
            Em = np.max(abs(y_train - predictions))
            # 计算emi
            emi = pow(predictions - y_train, 2) / pow(Em, 2)
            emi = np.array(emi).reshape(-1)
            # 计算误差率
            self.emrate.append(sum(self.weights * emi))
            # 计算学习器权重
            self.alpha.append(self.emrate[-1] / (1 - self.emrate[-1]))
            # 计算规范化因子
            ZM = sum(np.array(self.weights) * np.array([pow(self.alpha[-1], 1 - a) for a in emi]))
            # 更新权重
            self.weights = np.array(self.weights / ZM) * np.array([pow(self.alpha[-1], 1 - a) for a in emi])

            sleep(0.01)

    def Ada_predict(self, X_train, y_train, X_test, y_test, window):
        # 构建完毕，得到最终强回归器
        X_train, y_train, X_test, y_test, sx, sy = data_normalize(X_train, y_train, X_test, y_test, window)
        weight_alpha = sum([np.log(1 / b) for b in self.alpha])
        y_out = np.zeros((np.shape(y_test)))
        model = Sequential()
        model.add(LSTM(128, return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(1))
        model.compile(optimizer='adam',
                      loss='mean_squared_error')
        for i in range(self.model_num):
            model.fit(X_train, y_train, epochs=0, )
            model.load_weights(self.model[i])
            pre_y = model.predict(X_test)
            pre_y = sy.inverse_transform(pre_y)
            temp = np.log(1 / self.alpha[i]) / weight_alpha * pre_y
            y_out += temp
        y_test = sy.inverse_transform(y_test)
        print(f'AdaBoost集成模型:')
        print(f'R2: {r2_score(y_test, y_out):.2f}')
        print(f'MAE: {mean_absolute_error(y_test, y_out):.2f}')
        print(f'MSE: {mean_squared_error(y_test, y_out):.2f}')
        print('\n')
        return y_out
