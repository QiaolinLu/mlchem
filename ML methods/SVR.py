from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import LinearSVR
from sklearn.preprocessing import StandardScaler
from dataprocessing import *
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.svm import SVR
ss_x = StandardScaler()
ss_y = StandardScaler()

x_train = ss_x.fit_transform(X_vtrain)
x_test = ss_x.transform(X_test)
y_train = ss_y.fit_transform(y_BindingEnergy_vtrain)
y_test = ss_y.transform(y_BindingEnergy_test)



linear_svr = SVR(kernel='linear')
linear_svr.fit(x_train, y_train.ravel())
linear_svr_predict = linear_svr.predict(x_test)

poly_svr = SVR(kernel='poly')
poly_svr.fit(x_train, y_train.ravel())
poly_svr_predict = poly_svr.predict(x_test)

rbf_svr = SVR(kernel='rbf')
rbf_svr.fit(x_train, y_train.ravel())
rbf_svr_predict = rbf_svr.predict(x_test)


if __name__ == "__main__":
    print('The value of default measurement of linear SVR is', linear_svr.score(x_test, y_test))
    print('R-squared value of linear SVR is', r2_score(y_test, linear_svr_predict))
    print('The mean squared error of linear SVR is',
          mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict)))
    print('The mean absolute error of linear SVR is',
          mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(linear_svr_predict)))

    print('\nThe value of default measurement of poly SVR is', poly_svr.score(x_test, y_test))
    print('R-squared value of poly SVR is', r2_score(y_test, poly_svr_predict))
    print('The mean squared error of poly SVR is',
          mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_predict)))
    print('The mean absolute error of poly SVR is',
          mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(poly_svr_predict)))

    print('\nThe value of default measurement of rbf SVR is', rbf_svr.score(x_test, y_test))
    print('R-squared value of rbf SVR is', r2_score(y_test, rbf_svr_predict))
    print('The mean squared error of rbf SVR is',
          mean_squared_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_predict)))
    print('The mean absolute error of rbf SVR is',
          mean_absolute_error(ss_y.inverse_transform(y_test), ss_y.inverse_transform(rbf_svr_predict)))