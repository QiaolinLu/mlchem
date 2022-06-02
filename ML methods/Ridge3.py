from dataprocessing import *
from sklearn.linear_model import Lasso
from function import *
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

lasso = Lasso()
lasso.fit(X_vtrain, y_ActivationEnergy_vtrain)
#cv_results = cross_validate(RFR, X_vtrain, y_Q_vtrain, cv =10, scoring = "mean_sqaured_error")
scores = cross_val_score(lasso, X_vtrain, y_ActivationEnergy_vtrain, cv = 10, scoring="neg_mean_squared_error")
lasso_rmse_scores = np.sqrt(-scores)
y_ActivationEnergy_predict = lasso.predict(X_test)

if __name__ == "__main__":

    display_scores(lasso_rmse_scores)
    evaluation_indicators(y_ActivationEnergy_test, y_ActivationEnergy_predict)