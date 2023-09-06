import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from tree.randomForest import RandomForestClassifier
from tree.randomForest import RandomForestRegressor

np.random.seed(42)

########### RandomForestClassifier ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

for criteria in ['entropy', 'gini']:
    Classifier_RF = RandomForestClassifier(5, criterion = criteria, max_depth=5)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    fig1,fig2 = Classifier_RF.plot()
    #fig1.savefig("./Results/RF_C_fig1.png")
    fig2.savefig("./Results/RF_C_fig2.png")
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print("For class "+str(cls))
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))

########### RandomForestRegressor ###################

N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randn(N))

criteria='squared_error'
Regressor_RF = RandomForestRegressor(10, criterion = criteria, max_depth=10)
Regressor_RF.fit(X, y)
y_hat = Regressor_RF.predict(X)
# fig1, fig2 = Regressor_RF.plot()
# fig1.savefig("./Results/RF_R_fig1.png")
# fig2.savefig("./Results/RF_C_fig2.png")
print('Criteria :', criteria)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))
