from dataprocessing import *
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
X_train, X_val, y_train, y_val = train_test_split(X_vtrain, y_BindingEnergy_vtrain, test_size=0.2)
# 创建学习模型
knn = KNeighborsRegressor()
linear = LinearRegression()
ridge = Ridge()
lasso = Lasso()
decision = DecisionTreeRegressor()
svr = SVR()
# 训练模型
knn.fit(X_train, y_train)  # 学习率、惩罚项都封装好了
linear.fit(X_train, y_train)
ridge.fit(X_train, y_train)
lasso.fit(X_train, y_train)
decision.fit(X_train, y_train)
svr.fit(X_train, y_train)
# 预测数据
y_pre_knn = knn.predict(X_val)
y_pre_linear = linear.predict(X_val)
y_pre_ridge = ridge.predict(X_val)
y_pre_lasso = lasso.predict(X_val)
y_pre_decision = decision.predict(X_val)
y_pre_svr = svr.predict(X_val)


# 评分，R2 决定系数（拟合优度）。模型越好：r2→1；模型越差：r2→0
knn_score = r2_score(y_val, y_pre_knn)
linear_score = r2_score(y_val, y_pre_linear)
ridge_score = r2_score(y_val, y_pre_ridge)
lasso_score = r2_score(y_val, y_pre_lasso)
decision_score = r2_score(y_val, y_pre_decision)
svr_score = r2_score(y_val, y_pre_svr)
if __name__ == "__main__":
    # 绘图
    # KNN
    plt.plot(y_val, label='true')
    plt.plot(y_pre_knn, label='knn')
    plt.legend()

    # Linear
    plt.plot(y_val, label='true')
    plt.plot(y_pre_linear, label='linear')
    plt.legend()

    # Ridge
    plt.plot(y_val, label='true')
    plt.plot(y_pre_ridge, label='ridge')
    plt.legend()

    # Lasso
    plt.plot(y_val, label='true')
    plt.plot(y_pre_lasso, label='lasso')
    plt.legend()

    # Decision
    plt.plot(y_val, label='true')
    plt.plot(y_pre_decision, label='decision')
    plt.legend()

    # SVR
    plt.plot(y_val, label='true')
    plt.plot(y_pre_svr, label='svr')
    plt.legend()

    plt.show()
"""
决策树效果最好
"""