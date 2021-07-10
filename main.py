import numpy as np
from jqdatasdk import *
import seaborn as sns
import warnings
from predict_model import *
from sklearn.metrics import accuracy_score
from pre_plot import *

warnings.filterwarnings('ignore')

sns.set()
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 登录聚宽账户
auth('', '')

code = '000300.XSHG'  # 股票代码 000001.XSHG——上证指数；000300.XSHG——沪深300

predate = 365 * 5  # 总共获取数据数目（日度）

enddate = '2021-7-8'  # 获取数据截至日期

rate = 0.8  # 训练集占比

window = 30  # 用之前几日的数据预测之后一日的

factor_flag = 0  # 是否使用特征因子：1 是 0 否
# 特征因子包括：'MACD', 'DIFF', 'DEA', 'WVAD', 'OBV', 'RSI', 'MA10', 'MA20', 'MA60', 'BOLL_UB', 'BOLL_MB', 'BOLL_LB'

# 获取股票数据
stock_data, X_train, y_train, X_test, y_test, X_test_LSTM = get_data(code, predate, enddate, rate, window, factor_flag)

# LSTM预测
pre_LSTM, train_pre_LSTM = LSTM_pre(X_train, y_train, X_test_LSTM, y_test, window)
#
# LSTM_CNN预测
pre_LSTM_CNN, train_pre_LSTM_CNN = LSTM_CNN_pre(X_train, y_train, X_test_LSTM, y_test, window)

# CNN_LSTM预测
pre_CNN_LSTM, train_pre_CNN_LSTM = CNN_LSTM_pre(X_train, y_train, X_test_LSTM, y_test, window)

# 随机森林预测
pre_random, train_pre_random = random_forest(X_train, y_train, X_test, y_test)

# Adaboost预测
pre_Adaboost, train_pre_Adaboost = Adaboost(X_train, y_train, X_test, y_test)

# 使用岭回归集成各个模型
# 构建岭回归数据集
data = stock_data.filter(['close'])
train_Ridge = data[window:len(X_train)]
test_Ridge = data[len(X_train):]
train_Ridge = get_Ridge_data(train_Ridge, train_pre_LSTM, train_pre_LSTM_CNN, train_pre_CNN_LSTM, train_pre_random[window:],
                             train_pre_Adaboost[window:])
test_Ridge = get_Ridge_data(test_Ridge, pre_LSTM, pre_LSTM_CNN, pre_CNN_LSTM, pre_random, pre_Adaboost)
# 回归
predictions_Ridge = Ridge_pre(train_Ridge.iloc[:, 1:6], train_Ridge.iloc[:, 0], test_Ridge.iloc[:, 1:6], test_Ridge.iloc[:, 0])


# 使用Adoboost集成各个模型
Adaboost_all = Adaboost_integrate()
Adaboost_all.Ada_fit(X_train, y_train, X_test, y_test, window)
pre_Ada_all = Adaboost_all.Ada_predict(X_train, y_train, X_test_LSTM, y_test, window)

# 对比绘图
data = stock_data.filter(['close'])
train = data[:len(X_train)]
valid = data[len(X_train):]
true_trend = np.array(valid['close'] - valid['close'].shift(1))
true_trend = true_trend[1:len(true_trend)]
true_trend[true_trend > 0] = 1
true_trend[true_trend <= 0] = 0

# 画图显示结果
#
# LSTM预测结果
valid.loc[:, 'pre_LSTM'] = pre_LSTM
pre_trend = np.array(valid.loc[:, 'pre_LSTM'] - valid.loc[:, 'pre_LSTM'].shift(1))
pre_trend = pre_trend[1:len(pre_trend)]
pre_trend[pre_trend > 0] = 1
pre_trend[pre_trend <= 0] = 0
print(f'LSTM Accuracy: {accuracy_score(true_trend, pre_trend):.2f}')
plot_LSTM(code, train, valid)

# LSTM_CNN预测结果
valid.loc[:, 'pre_LSTM_CNN'] = pre_LSTM_CNN
pre_trend = np.array(valid.loc[:, 'pre_LSTM_CNN'] - valid.loc[:, 'pre_LSTM_CNN'].shift(1))
pre_trend = pre_trend[1:len(pre_trend)]
pre_trend[pre_trend > 0] = 1
pre_trend[pre_trend <= 0] = 0
print(f'LSTM_CNN Accuracy: {accuracy_score(true_trend, pre_trend):.2f}')
plot_LSTM_CNN(code, train, valid)

# CNN_LSTM预测结果
valid.loc[:, 'pre_CNN_LSTM'] = pre_CNN_LSTM
pre_trend = np.array(valid.loc[:, 'pre_CNN_LSTM'] - valid.loc[:, 'pre_CNN_LSTM'].shift(1))
pre_trend = pre_trend[1:len(pre_trend)]
pre_trend[pre_trend > 0] = 1
pre_trend[pre_trend <= 0] = 0
print(f'CNN_LSTM Accuracy: {accuracy_score(true_trend, pre_trend):.2f}')
plot_CNN_LSTM(code, train, valid)

# 随机森林预测结果
valid.loc[:, 'pre_random'] = pre_random
pre_trend = np.array(valid.loc[:, 'pre_random'] - valid.loc[:, 'pre_random'].shift(1))
pre_trend = pre_trend[1:len(pre_trend)]
pre_trend[pre_trend > 0] = 1
pre_trend[pre_trend <= 0] = 0
print(f'Random_forest Accuracy: {accuracy_score(true_trend, pre_trend):.2f}')
plot_RF(code, train, valid)

# Adaboost预测结果
valid.loc[:, 'pre_Adaboost'] = pre_Adaboost
pre_trend = np.array(valid.loc[:, 'pre_Adaboost'] - valid.loc[:, 'pre_Adaboost'].shift(1))
pre_trend = pre_trend[1:len(pre_trend)]
pre_trend[pre_trend > 0] = 1
pre_trend[pre_trend <= 0] = 0
print(f'Adaboost Accuracy: {accuracy_score(true_trend, pre_trend):.2f}')
plot_Adaboost(code, train, valid)

# 岭回归预测结果
valid.loc[:, 'predictions_Ridge'] = predictions_Ridge
pre_trend = np.array(valid.loc[:, 'predictions_Ridge'] - valid.loc[:, 'predictions_Ridge'].shift(1))
pre_trend = pre_trend[1:len(pre_trend)]
pre_trend[pre_trend > 0] = 1
pre_trend[pre_trend <= 0] = 0
print(f'Ridge Accuracy: {accuracy_score(true_trend, pre_trend):.2f}')
plot_Ridge(code, train, valid)

# Adaboost集成预测结果
valid.loc[:, 'pre_Ada_all'] = pre_Ada_all
pre_trend = np.array(valid.loc[:, 'pre_Ada_all'] - valid.loc[:, 'pre_Ada_all'].shift(1))
pre_trend = pre_trend[1:len(pre_trend)]
pre_trend[pre_trend > 0] = 1
pre_trend[pre_trend <= 0] = 0
print(f'Adaboost_LSTM Accuracy: {accuracy_score(true_trend, pre_trend):.2f}')
plot_Ada_all(code, train, valid)

plot_all(code, train, valid)