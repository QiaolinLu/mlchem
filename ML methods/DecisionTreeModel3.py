from dataprocessing import *
from function import *
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from sklearn.ensemble import RandomForestRegressor
regressor = DecisionTreeRegressor()
regressor.fit(X_vtrain, y_Q_vtrain)
scores = cross_val_score(regressor, X_vtrain, y_Q_vtrain, cv = 10, scoring="neg_mean_squared_error")
tree_rmse_scores = np.sqrt(-scores)
y_Q_predict = regressor.predict(X_test)


if __name__ == "__main__":
    display_scores(tree_rmse_scores)
    evaluation_indicators(y_Q_test, y_Q_predict)