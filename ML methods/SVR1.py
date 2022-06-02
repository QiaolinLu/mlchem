from dataprocessing import *
from sklearn.svm import SVR
from function import *
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

svr = SVR(kernel="linear", C=10.0)
svr.fit(X_vtrain, y_BindingEnergy_vtrain)
#cv_results = cross_validate(RFR, X_vtrain, y_Q_vtrain, cv =10, scoring = "mean_sqaured_error")
scores = cross_val_score(svr, X_vtrain, y_BindingEnergy_vtrain, cv = 10, scoring="neg_mean_squared_error")
svr_rmse_scores = np.sqrt(-scores)
y_BindingEnergy_predict = svr.predict(X_test)

if __name__ == "__main__":

    display_scores(svr_rmse_scores)
    evaluation_indicators(y_BindingEnergy_test, y_BindingEnergy_predict)