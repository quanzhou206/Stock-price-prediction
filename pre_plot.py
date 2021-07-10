import matplotlib.pyplot as plt


def plot_LSTM(code, train, valid):
    plt.figure(figsize=(16, 8))
    plt.title(code)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('close Price', fontsize=18)
    plt.plot(train)
    plt.plot(valid[['close', 'pre_LSTM']])
    plt.legend(['Train', 'Test', 'LSTM'])
    plt.title('Stock ' + code + '  LSTM Price forecast')
    plt.show()


def plot_LSTM_CNN(code, train, valid):
    plt.figure(figsize=(16, 8))
    plt.title(code)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('close Price', fontsize=18)
    plt.plot(train)
    plt.plot(valid[['close', 'pre_LSTM_CNN']])
    plt.legend(['Train', 'Test', 'LSTM_CNN'])
    plt.title('Stock ' + code + '  LSTM_CNN Price forecast')
    plt.show()


def plot_CNN_LSTM(code, train, valid):
    plt.figure(figsize=(16, 8))
    plt.title(code)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('close Price', fontsize=18)
    plt.plot(train)
    plt.plot(valid[['close', 'pre_CNN_LSTM']])
    plt.legend(['Train', 'Test', 'CNN_LSTM'])
    plt.title('Stock ' + code + '  CNN_LSTM Price forecast')
    plt.show()


def plot_RF(code, train, valid):
    plt.figure(figsize=(16, 8))
    plt.title(code)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('close Price', fontsize=18)
    plt.plot(train)
    plt.plot(valid[['close', 'pre_random']])
    plt.legend(['Train', 'Test', 'Random_forest'])
    plt.title('Stock ' + code + '  Random_forest Price forecast')
    plt.show()


def plot_Adaboost(code, train, valid):
    plt.figure(figsize=(16, 8))
    plt.title(code)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('close Price', fontsize=18)
    plt.plot(train)
    plt.plot(valid[['close', 'pre_Adaboost']])
    plt.legend(['Train', 'Test', 'Adaboost'])
    plt.title('Stock ' + code + '  Adaboost Price forecast')
    plt.show()


def plot_Ridge(code, train, valid):
    plt.figure(figsize=(16, 8))
    plt.title(code)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('close Price', fontsize=18)
    plt.plot(train)
    plt.plot(valid[['close', 'predictions_Ridge']])
    plt.legend(['Train', 'Test', 'Ridge'])
    plt.title('Stock ' + code + '  Ridge Price forecast')
    plt.show()


def plot_all(code, train, valid):
    plt.figure(figsize=(16, 8))
    plt.title(code)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('close Price', fontsize=18)
    plt.plot(train)
    plt.plot(
        valid[['close', 'pre_LSTM', 'pre_LSTM_CNN', 'pre_CNN_LSTM', 'pre_random', 'pre_Adaboost', 'predictions_Ridge','pre_Ada_all']])
    plt.legend(['Train', 'Test', 'Ridge', 'LSTM_CNN', 'CNN_LSTM', 'random', 'Adaboost', 'Ridge','Adaboost_LSTM'])
    plt.title('Stock ' + code + '  Price forecast')
    plt.show()

def plot_Ada_all(code, train, valid):
    plt.figure(figsize=(16, 8))
    plt.title(code)
    plt.xlabel('Date', fontsize=18)
    plt.ylabel('close Price', fontsize=18)
    plt.plot(train)
    plt.plot(valid[['close', 'pre_Ada_all']])
    plt.legend(['Train', 'Test', 'Adaboost_integrate'])
    plt.title('Stock ' + code + '  Adaboost_LSTM Price forecast')
    plt.show()
