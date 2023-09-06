import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *

from ensemble.gradientBoosted import GradientBoostedRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.datasets import make_regression
import matplotlib.pyplot as plt
   

# Or use sklearn decision tree

########### GradientBoostedClassifier ###################

X, y= make_regression(
       n_features=3,
       n_informative=3,
       noise=10,
       tail_strength=10,
       random_state=42,
   )



X= pd.DataFrame(X)
y= pd.Series(y)

criteria='squared_error'
lr = 0.1                #learning rate
n_estimators=100        
tree = DecisionTreeRegressor(criterion=criteria, max_depth=4)
GD_tree = GradientBoostedRegressor(base_estimator=tree , n_estimators= n_estimators, learning_rate=lr)
GD_tree.fit(X, y)
y_hat = GD_tree.predict(X)

print('Criteria :', criteria)
print('RMSE: ', rmse(y_hat, y))
print('MAE: ', mae(y_hat, y))