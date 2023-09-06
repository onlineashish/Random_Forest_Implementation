import numpy as np
import pandas as pd
import copy
import math
from typing import Union
import matplotlib.pyplot as plt

from metrics import *


class GradientBoostedRegressor:
    def __init__(self, base_estimator, n_estimators=100, learning_rate=0.1):  # Optional Arguments: Type of estimator
        """
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        :param learning_rate: The learning rate shrinks the contribution of each tree by `learning_rate`.
        """
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.lr = learning_rate
        self.my_model=None
        
        

    def fit(self, X, y):
        """
        Function to train and construct the GradientBoostedRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        X = pd.DataFrame(X)
        y = pd.Series(y)
        self.meen = y.mean()
        y_hat = pd.Series([y.mean()]* len(y))
        residuals = y - y_hat
        for i in range(self.n_estimators):
            tree=copy.deepcopy(self.base_estimator)
            tree.fit(X,residuals)
            self.my_model=(tree)
            res_cur = tree.predict(X)
            y_hat_new = y_hat + (self.lr * res_cur)
            #if no further improvement
            if (y_hat_new == y_hat).all():
                break
            y_hat = y_hat_new
            residuals = y - y_hat
        self.y_haat = y_hat.copy()
        
        pass

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hat = self.my_model.predict(X)
        y_hat = y_hat + self.y_haat
        return y_hat
