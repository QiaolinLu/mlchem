import matplotlib.pyplot as plt
from dataprocessing import *
import xgboost
import graphviz
from function import *
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

xgb = xgboost.XGBRegressor()
xgb.fit(X_vtrain, y_ActivationEnergy_vtrain)
#cv_results = cross_validate(RFR, X_vtrain, y_Q_vtrain, cv =10, scoring = "mean_sqaured_error")
scores = cross_val_score(xgb, X_vtrain, y_ActivationEnergy_vtrain, cv = 10, scoring="neg_mean_squared_error")
xgb_rmse_scores = np.sqrt(-scores)
y_ActivationEnergy_predict = xgb.predict(X_test)

if __name__ == "__main__":

    display_scores(xgb_rmse_scores)
    evaluation_indicators(y_ActivationEnergy_test, y_ActivationEnergy_predict)
    xgboost.plot_tree(xgb)

