from dataprocessing import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from function import *
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor()
scores = cross_val_score(gbrt, X_vtrain, y_Q_vtrain, cv=10, scoring="neg_mean_squared_error")
gbrt.fit(X_vtrain, y_Q_vtrain)
gbrt_rmse_scores = np.sqrt(-scores)
y_Q_predict = gbrt.predict(X_test)

if __name__ == "__main__":
    display_scores(gbrt_rmse_scores)
    print("Q_true:", y_Q_test)
    print("Predict:", y_Q_predict)
    #test_mse = mean_squared_error(y_Q_predict, y_Q_test)
    #test_rmse = np.sqrt(test_mse)
    #display_scores(test_rmse)
    evaluation_indicators(y_Q_predict, y_Q_test)
