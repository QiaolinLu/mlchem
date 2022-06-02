from dataprocessing import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from function import *
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

RFR = RandomForestRegressor()
RFR.fit(X_vtrain, y_ActivationEnergy_vtrain)
#cv_results = cross_validate(RFR, X_vtrain, y_ActivationEnergy_vtrain, cv =10, scoring = "mean_sqaured_error")
scores = cross_val_score(RFR, X_vtrain, y_ActivationEnergy_vtrain, cv = 10, scoring="neg_mean_squared_error")
rfr_rmse_scores = np.sqrt(-scores)
y_ActivationEnergy_predict = RFR.predict(X_test)

if __name__ == "__main__":
    display_scores(rfr_rmse_scores)
    evaluation_indicators(y_ActivationEnergy_test, y_ActivationEnergy_predict)
