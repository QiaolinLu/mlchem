from dataprocessing import *
from sklearn.neighbors import KNeighborsRegressor
from function import *
from sklearn.model_selection import cross_val_score, cross_validate
import numpy as np

knn = KNeighborsRegressor()
knn.fit(X_vtrain, y_Q_vtrain)
#cv_results = cross_validate(RFR, X_vtrain, y_Q_vtrain, cv =10, scoring = "mean_sqaured_error")
scores = cross_val_score(knn, X_vtrain, y_Q_vtrain, cv = 10, scoring="neg_mean_squared_error")
knn_rmse_scores = np.sqrt(-scores)
y_Q_predict = knn.predict(X_test)

if __name__ == "__main__":

    display_scores(knn_rmse_scores)
    evaluation_indicators(y_Q_test, y_Q_predict)