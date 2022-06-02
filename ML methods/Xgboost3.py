from dataprocessing import *
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from function import *
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

xgb = xgb.XGBRegressor()
xgb.fit(X_vtrain, y_Q_vtrain)
#cv_results = cross_validate(RFR, X_vtrain, y_Q_vtrain, cv =10, scoring = "mean_sqaured_error")
scores = cross_val_score(xgb, X_vtrain, y_Q_vtrain, cv = 10, scoring="neg_mean_squared_error")
xgb_rmse_scores = np.sqrt(-scores)
y_Q_predict = xgb.predict(X_test)

if __name__ == "__main__":

    display_scores(xgb_rmse_scores)
    evaluation_indicators(y_Q_test, y_Q_predict)