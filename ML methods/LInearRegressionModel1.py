from sklearn.linear_model import LinearRegression
from dataprocessingv2 import *
from function import *
import numpy as np
np.set_printoptions(suppress=True)

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
lin_reg.fit(X_vtrain, y_BindingEnergy_vtrain)
scores = cross_val_score(lin_reg, X_vtrain, y_BindingEnergy_vtrain, cv = 20, scoring="neg_mean_squared_error")
lin_rmse_scores = np.sqrt(-scores)
y_BindingEnergy_predict = lin_reg.predict(X_test)

if __name__ == "__main__":
    display_scores(lin_rmse_scores)
    evaluation_indicators(y_BindingEnergy_test, y_BindingEnergy_predict)
    print("model_coef_:\n", lin_reg.coef_)
    print("model_intercept_:\n", lin_reg.intercept_)