from keras import layers
from keras import models
from dataprocessing import *
import numpy as np
from function import *
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from function import *
# create model
model = models.Sequential()
model.add(layers.BatchNormalization(input_dim=9))
model.add(layers.Dropout(0.3))
# model.add(layers.Dense(7, kernel_initializer = 'normal', activation = 'relu'))
model.add(layers.Dense(1))
# compile model
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])
if __name__ == "__main__":
    k = 10
    num = len(X_vtrain) // k
    num_epochs = 20
    all_list = []
    for i in range(k):
        print('proccesing #', i)
        val_data = X_vtrain[i * num:(i + 1) * num]  # 从训练集中提取出验证的数据部分
        val_target = y_BindingEnergy_vtrain[i * num:(i + 1) * num]  # 从训练集中提取出验证的标签部分（房价）

        par_data = np.concatenate(  # 把训练数据的其他部分粘合在一起
            [X_vtrain[:i * num],
             X_vtrain[(i + 1) * num:]],
            axis=0
        )
        par_target = np.concatenate(  # 把训练标签的其他部分粘合在一起
            [y_BindingEnergy_vtrain[:i * num],
             y_BindingEnergy_vtrain[(i + 1) * num:]],
            axis=0
        )
        his = model.fit(par_data, par_target, epochs=num_epochs,
                        batch_size=1, validation_data=(val_data, val_target))
        history = his.history['mse']
        all_list.append(history)

    ave_list = [np.mean([x[i] for x in all_list]) for i in range(num_epochs)]
    plt.plot(range(1, len(ave_list) + 1), ave_list)
    plt.xlabel('epochs')
    plt.ylabel('train mse')
    plt.show()
    evaluation_indicators(y_BindingEnergy_test, model.predict(X_test).flatten())