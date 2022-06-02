from dataprocessing import *
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from function import *
from sklearn.ensemble import GradientBoostingRegressor

gbrt = GradientBoostingRegressor()
scores = cross_val_score(gbrt, X_vtrain, y_BindingEnergy_vtrain, cv=10, scoring="neg_mean_squared_error")
gbrt.fit(X_vtrain, y_BindingEnergy_vtrain)
gbrt_rmse_scores = np.sqrt(-scores)
y_BindingEnergy_predict = gbrt.predict(X_test)

if __name__ == "__main__":
    display_scores(gbrt_rmse_scores)
    evaluation_indicators(y_BindingEnergy_test, y_BindingEnergy_predict)