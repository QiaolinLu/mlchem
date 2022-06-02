from dataprocessing import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeRegressor
import numpy as np
from function import *
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from sklearn.ensemble import RandomForestRegressor
regressor = DecisionTreeRegressor(random_state=42)
regressor.fit(X_vtrain, y_BindingEnergy_vtrain)
scores = cross_val_score(regressor, X_vtrain, y_BindingEnergy_vtrain, cv = 10, scoring="neg_mean_squared_error")
tree_rmse_scores = np.sqrt(-scores)
y_BindingEnergy_predict = regressor.predict(X_test)


if __name__ == "__main__":

    display_scores(tree_rmse_scores)
    #test_mse = mean_squared_error(y_BindingEnergy_predict, y_BindingEnergy_test)
    #rmse = np.sqrt(test_mse)
    #display_scores(rmse)
    evaluation_indicators(y_BindingEnergy_test, y_BindingEnergy_predict)


