import matplotlib.pyplot as plt
import numpy as np
import math
from random import gauss
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.metrics import r2_score
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
## add noise to the original data to expanda the dataset
"""
def gaussian_noise(inputs):
    mu = 0
    sigma = 0.001
    for i in range(len(inputs)):
        inputs[i] += gauss(mu, sigma)
    return inputs
def fakedata_produce(df):
    for index in range(0, len(df)):
        for i in range(0, 100):
            df.loc[len(df) + index] = gaussian_noise(df.iloc[index])
    return df
"""
"""
from random import gauss
 # about the parameters of sigma, mu
def gauss_noise2(inputs, mu, sigma):
    for i in range(len(inputs)):
        inputs[i] += gauss(mu[i], sigma[i])
    return inputs
def fake_data(df, mu, sigma):
    for index in range(len(df)):
        for i in range(0, 10):
            df.loc[len(df) + index] = gauss_noise2(df.iloc[index].values, mu, sigma)
    return df
"""
from random import gauss
def fake_data(df, alpha, num):
    def gauss_noise2(inputs, mu, sigma):
       for i in range(len(inputs)):
        inputs[i] += gauss(mu[i], sigma[i])
        return inputs
    sigma = alpha * df.std().values
    mu = [0] * df.shape[1]
    l = len(df)
    for index in range(l):
        for i in range(0, num):
            df.loc[l + index*num+i] = gauss_noise2(df.iloc[index].values, mu, sigma)
    return df.drop(df.head(l).index)
def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size = 0.2)
    train_errors, val_errors = [], []
    for m in range(1,len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train[:m], y_train_predict))
        val_errors.append(mean_squared_error(y_val[:m], y_val_predict))
    plt.plot(np.sqrt(train_errors), "r+", linewidth = 2, label = "train")
    plt.plot(np.sqrt(val_errors), 'b-', linewidth = 3, label = "val")
def display_scores(scores):
    print("Scores: ", scores)
    print("Mean:", scores.mean())
    print("Standard deviation: ", scores.std())

# 误差
def get_average(records):
    """
    均值
    """
    return sum(records) / len(records)
def get_variance(records):
    """
    方差
    """
    average = get_average(records)
    return sum([(x-average)**2 for x in records]) / len(records)
def get_standard_deviation(records):
    """
    标准差
    """
    variance = get_variance(records)
    return math.sqrt(variance)
def get_rms(records):
    """
    均方根值 反映的是有效值而不是平均值
    """
    return math.sqrt(sum([x ** 2 for x in records]) / len(records))
def get_mse(records_real, records_predict):
    """
    均方误差 估计值与真值 偏差
    """
    if len(records_real) == len(records_predict):
        return sum([(x - y) ** 2 for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None
def get_rmse(records_real, records_predict):
    """
    均方根误差：是均方误差的算术平方根
    """
    mse = get_mse(records_real, records_predict)
    if mse:
        return math.sqrt(mse)
    else:
        return None
def get_mae(records_real, records_predict):
    """
    平均绝对误差
    """
    if len(records_real) == len(records_predict):
        return sum([abs(x - y) for x, y in zip(records_real, records_predict)]) / len(records_real)
    else:
        return None
def get_mape(records_real, records_predict):
    """
    平均绝对百分误差
    :param records_real:
    :param records_predict:
    :return:
    """
    if len(records_real) == len(records_predict):
        return mean_absolute_percentage_error(records_real, records_predict)
    else:
        return None
def get_r2score(records_real, records_predict):
    """
    r2方法
    :param records_real:
    :param records_predict:
    :return:
    """
    if len(records_real) == len(records_predict):
        return r2_score(records_real, records_predict)
    else:
        return None
def evaluation_indicators(records_real, records_predict):

    print("real value:\n", records_real)
    print("predict value:\n", records_predict)
    print("mae:\n", get_mae(records_real, records_predict))
    print("mse:\n", get_mse(records_real, records_predict))
    print("rmse:\n", get_rmse(records_real, records_predict))
    print("mape:\n", get_mape(records_real, records_predict))
    print("r2 score:\n", get_r2score(records_real, records_predict))





