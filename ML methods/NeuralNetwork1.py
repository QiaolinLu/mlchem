from dataprocessing import *
from keras.models import Model
from keras import layers
import numpy as np
from sklearn.metrics import mean_squared_error
import keras
import os
num_epochs = 50

k = 10
num = len(X_vtrain)//k
X_Inputs = layers.Input(shape=(9, ))
X = layers.BatchNormalization()(X_Inputs)
X = layers.Dropout(0.2)(X)
BE = layers.Dense(10)(X)
BE_prediction = layers.Dense(1, name = 'binding_energy_prediction')(BE)
AE = layers.Dense(10)(X)
AE_prediction = layers.Dense(1, name = 'activation_energy_prediction')(AE)
Q = layers.Dense(10)(X)
Q_prediction = layers.Dense(1, name = 'Q_prediction')(Q)
model_reg = Model(X_Inputs,
                  [BE_prediction, AE_prediction, Q_prediction])
model_reg.compile(optimizer = 'adam', loss = {'binding_energy_prediction': 'mse',
                                                 'activation_energy_prediction': 'mse',
                                                 'Q_prediction': 'mse'},
                  loss_weights = [1, 1, 1])

if __name__ == "__main__":
    all_list = []
    all_scores = []
    for i in range(k):
        print("Processing #", i)
        val_data = X_vtrain[i*num:(i+1)*num]
        val_BE = y_BindingEnergy_vtrain[i * num:(i + 1) * num]
        val_AE = y_ActivationEnergy_vtrain[i * num:(i + 1) * num]
        val_Q = y_Q_vtrain[i * num:(i + 1) * num]
        val_target = [val_BE, val_AE, val_Q]

        par_data = np.concatenate([X_vtrain[:i * num], X_vtrain[(i + 1) * num:]], axis=0)
        par_BE = np.concatenate([y_BindingEnergy_vtrain[:i * num], y_BindingEnergy_vtrain[(i + 1) * num:]], axis=0)
        par_AE = np.concatenate([y_ActivationEnergy_vtrain[:i * num], y_ActivationEnergy_vtrain[(i + 1) * num:]],
                                axis=0)
        par_Q = np.concatenate([y_Q_vtrain[:i * num], y_Q_vtrain[(i + 1) * num:]], axis=0)
        par_target = [par_BE, par_AE, par_Q]
        his = model_reg.fit(par_data, par_target, validation_data=(val_data, val_target), epochs=50, batch_size=5,
                            verbose=0)
        #val_mae, val_mse = model_reg.evaluate(val_data, par_target, verbose = 0)
        #all_scores.append(val_mae)
    ave_list = [np.mean([x[i] for x in all_list]) for i in range(num_epochs)]
    #mae_mean = np.mean(all_scores)
    #print("mae_mean", mae_mean)
    print("BE : \n", model_reg.predict(X_test)[0].flatten())
    print("BindingEnergy_True: \n", y_BindingEnergy_test)
    net_mse_BE = mean_squared_error(model_reg.predict(X_test)[0].flatten(), y_ActivationEnergy_test)
    print("BE rmse: \n", np.sqrt(net_mse_BE))
    print("AE : \n", model_reg.predict(X_test)[1].flatten())
    print("ActivationEnergy_True: \n", y_ActivationEnergy_test)
    net_mse_AE = mean_squared_error(model_reg.predict(X_test)[1].flatten(), y_ActivationEnergy_test)
    print("AE rmse: \n", np.sqrt(net_mse_AE))
    print("Q : \n", model_reg.predict(X_test)[2].flatten())
    print("Q_True: \n", y_Q_test)
    net_mse_Q = mean_squared_error(model_reg.predict(X_test)[1].flatten(), y_Q_test)
    print("Q rmse: \n", np.sqrt(net_mse_Q))
