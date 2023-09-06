import pandas as pd
import numpy as np
import random
import operator
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
import time
from multiprocessing import Pool
from joblib import Parallel, delayed
import os
import concurrent.futures

class BaggingClassifier():
    def __init__(self, base_estimator, n_estimators=100, n_jobs = os.cpu_count()):
        '''
        :param base_estimator: The base estimator model instance from which the bagged ensemble is built (e.g., DecisionTree(), LinearRegression()).
                               You can pass the object of the estimator class
        :param n_estimators: The number of estimators/models in ensemble.
        '''
        self.n_estimators = n_estimators
        self.classifiers = [copy.deepcopy(base_estimator) for i in range(n_estimators)]
        self.X = [None for i in range(n_estimators)]
        self.Y = [None for i in range(n_estimators)]
        self.n_jobs = n_jobs


    def fit(self, X, y):
        """
        Function to train and construct the BaggingClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        assert(X.shape[0]==y.size)  

        data = pd.concat([X,y],axis=1)
        

        if (self.n_jobs!= None):
            lis_x =[]
            lis_y = []
            for round in range(self.n_estimators):
                data1 = data.sample(n=y.size, replace=True).reset_index(drop=True)
                X1 = data1.iloc[:, :-1]
                y1 = data1.iloc[:, -1]
                self.X[round] = X1
                self.Y[round] = y1
                lis_x.append(X1)
                lis_y.append(y1)
                # self.classifiers[round].fit(X1,y1)

            

            with concurrent.futures.ProcessPoolExecutor(self.n_jobs) as executor:
                result = list(executor.map(self.classifiers[0].fit, lis_x, lis_y))

            self.classifiers = result



        #if no parallel processing
        

        for round in range(self.n_estimators):
            data1 = data.sample(n=y.size, replace=True).reset_index(drop=True)
            X1 = data1.iloc[:, :-1]
            y1 = data1.iloc[:, -1]
            self.X[round] = X1
            self.Y[round] = y1
            self.classifiers[round].fit(X1,y1)

        

    def predict(self, X):
        """
        Funtion to run the BaggingClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_hats = list()
        for clf in self.classifiers:
            y_hats.append(clf.predict(X))
        
        y_hat = list()
        for i in range(X.shape[0]):
            predictions = dict()
            for pred in y_hats:
                if(pred[i] in predictions):
                    predictions[pred[i]] += 1
                else:
                    predictions[pred[i]] = 1
            y_hat.append(max(predictions.items(), key=operator.itemgetter(1))[0])
        
        return pd.Series(y_hat)


    def plot(self):
        """
        Function to plot the decision surface for BaggingClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns and should look similar to slide #16 of lecture
        The title of each of the estimator should be iteration number

        Figure 2 should also create a decision surface by combining the individual estimators and should look similar to slide #16 of lecture

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]

        """

        h=0.02
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])

        fig1 = plt.figure(figsize=(5*self.n_estimators,5))

        for i in range(self.n_estimators):
            plt.subplot(1,self.n_estimators,i+1)
            plt.scatter(self.X[i].iloc[:,0], self.X[i].iloc[:,1], c=self.Y[i], cmap=cm_bright, edgecolors='k')
            plt.xlabel(str(self.X[i].columns[0]))
            plt.ylabel(str(self.X[i].columns[1]))
            plt.title("Round "+str(i+1))
        
        plt.show()

        fig2 = plt.figure(figsize=(5*self.n_estimators,5))
        
        for i in range(self.n_estimators):
            x_min, x_max = self.X[i].iloc[:, 0].min() - .5, self.X[i].iloc[:, 0].max() + .5
            y_min, y_max = self.X[i].iloc[:, 1].min() - .5, self.X[i].iloc[:, 1].max() + .5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            
            
            Z = np.array(self.classifiers[i].predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], 
                            columns=self.X[i].columns)))
            
            Z = Z.reshape(xx.shape)

            plt.subplot(1,self.n_estimators,i+1)
            plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            plt.scatter(self.X[i].iloc[:,0], self.X[i].iloc[:,1], c=self.Y[i], cmap=cm_bright, edgecolors='k')
            
            plt.xlabel(str(self.X[i].columns[0]))
            plt.ylabel(str(self.X[i].columns[1]))
            plt.title("Round "+str(i+1))
        
        plt.show()

        return [fig1, fig2]