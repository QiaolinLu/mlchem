from sklearn.linear_model import LinearRegression
from dataprocessing import *
from function import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score

lin_reg = LinearRegression()
lin_reg.fit(X_vtrain, y_Q_vtrain)
scores = cross_val_score(lin_reg, X_vtrain, y_Q_vtrain, cv = 10, scoring="neg_mean_squared_error")
lin_rmse_scores = np.sqrt(-scores)
y_Q_predict = lin_reg.predict(X_test)

if __name__ == "__main__":
    display_scores(lin_rmse_scores)
    evaluation_indicators(y_Q_test, y_Q_predict)