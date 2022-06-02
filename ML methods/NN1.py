from keras import layers
from keras import models
from dataprocessing import *
from sklearn.metrics import *
from function import *
from sklearn.model_selection import cross_val_score

model = models.Sequential()
model.add(layers.BatchNormalization(input_dim=9))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(1))
model.compile(optimizer='sgd', loss= 'mse', metrics=['mse'])


model.fit(X_vtrain, y_BindingEnergy_vtrain)
nn_scores = cross_val_score(model, X_vtrain, y_BindingEnergy_vtrain, cv = 10, scoring="neg_mean_squared_error")
nn_rmse_scores = np.sqrt(-nn_scores)

y_BindingEnergy_predict = model.predict(X_test).flatten()

if __name__ == "__main__":
    display_scores(nn_rmse_scores)
    evaluation_indicators(y_BindingEnergy_test, y_BindingEnergy_predict)