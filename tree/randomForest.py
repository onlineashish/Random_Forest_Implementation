from .base import DecisionTree
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor
import numpy as np
import pandas as pd
from statistics import mean
from sklearn.tree import export_graphviz
from subprocess import call
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy
from sklearn.decomposition import PCA
# import Image





class RandomForestClassifier():
    def __init__(self, n_estimators=100, criterion='gini', max_depth=None):
        '''
        :param estimators: DecisionTree
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
       

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        self.trees = []
        n_features = X.shape[1]
        self.featureidx = []
        self.subXs = []
        m = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            clf = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth)
            idx = np.random.choice(range(n_features), size=m, replace=True)
            X_sub = X[idx]
            self.subXs.append(copy.deepcopy(X_sub))
            self.featureidx.append(idx)
            clf.fit(X_sub, y)
            self.trees.append(clf)
               
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        """
        Funtion to run the RandomForestClassifier on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        def most_element(l):
            counts = [[l.count(e),e] for e in l]
            counts.sort(key=lambda x:x[0], reverse=True)
            return counts[0][1]
            
        y_pred=[]
        for i in X.index.values:
            X_test = X[X.index==i]
            preds=[]
            for j in range(self.n_estimators):
                X_tt = X_test[self.featureidx[j]]
                pred = self.trees[j].predict(X_tt)
                preds.append(pred[0])
            y_pred.append(most_element(preds))

        return pd.Series(y_pred)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface for each estimator

        3. Creates a figure showing the combined decision surface

        """
        h = .02 # gaps in the grid
        i=1
        fig1 = plt.figure(figsize=(27, 9))
        subXs = self.subXs
        X = self.X
        y = self.y
        x_min, x_max = X[X.columns[0]].min() - .5, X[X.columns[0]].max() + .5
        y_min, y_max = X[X.columns[1]].min() - .5, X[X.columns[1]].max() + .5
        # z_min, z_max = X[X.columns[2]].min() - .5, X[X.columns[2]].max() + .5
        # a_min, a_max = X[X.columns[3]].min() - .5, X[X.columns[3]].max() + .5

        # xx, yy, zz, aa = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h), np.arange(a_min, a_max, h))
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
       
        for treeo in self.trees:
            ax = plt.subplot(1, len(self.trees) , i)
            X = pd.DataFrame(subXs[i-1])
            y = self.y
            # x_min, x_max = X[X.columns[0]].min() - .5, X[X.columns[0]].max() + .5
            # y_min, y_max = X[X.columns[1]].min() - .5, X[X.columns[1]].max() + .5
            # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = np.array(treeo.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)))
            # print(Z[12])
            Z = Z.reshape(xx.shape)
            X = self.X
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            ax.scatter(X[X.columns[0]][:30], X[X.columns[1]][:30], c=y, cmap=cm_bright, edgecolors='k')
            # size=[1000*w for w in self.weights[i-1]]
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel(str(X.columns[0]))
            ax.set_ylabel(str(X.columns[1]))
            plt.title("Estimator "+str(i))
            i+=1
        
        #plt.clf()
        X = self.X
        h = 0.5
        x_min, x_max = X[X.columns[0]].min() - .5, X[X.columns[0]].max() + .5
        y_min, y_max = X[X.columns[1]].min() - .5, X[X.columns[1]].max() + .5

        # Creating Mess grid.
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),np.arange(y_min, y_max, h))

        # Identifying if we have more than 2 features or not, because we can only plot the surface for two features.
        if len(self.X.columns) > 2:

            # If we have more than two features, we will take the mean value of the remaining features.
            remainder = pd.DataFrame()

            # Taking mean of every column after 2nd column.
            for j in range(2,len(self.X.columns)):
                mean = self.X[X.columns[j]].mean()

                # We make fill the space with the mean of that columns.
                remainder[j] = np.ones(len(np.c_[xx.ravel(), yy.ravel()]))*mean

            # Predicting the values for the Mess grid.
            Z = np.array(self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel(),remainder],columns=X.columns)))

        else:
            # Predicting the values for the Mess grid.
            Z = np.array(self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()],columns=X.columns)))

        # Reshaping Z to xx's shape so that we can plot it.
        Z = Z.reshape(xx.shape)

        # Setting labels/titles and creating the plot for 'figure 2'.
        fig2 = plt.figure(figsize=(9,9))
        
        plt.contourf(xx, yy, Z, alpha=0.5)
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        
        plt.scatter(self.X[0],self.X[1],c=self.y,edgecolors='k')
            
        

        return [fig1,fig2]
        



class RandomForestRegressor():
    def __init__(self, n_estimators=100, criterion='mse', max_depth=None):
        '''
        :param n_estimators: The number of trees in the forest.
        :param criterion: The function to measure the quality of a split.
        :param max_depth: The maximum depth of the tree.
        '''
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth

        pass

    def fit(self, X, y):
        """
        Function to train and construct the RandomForestRegressor
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        trees = []
        n_features = X.shape[1]
        featureidx = []
        m = int(np.sqrt(n_features))
        for i in range(self.n_estimators):
            clf = DecisionTreeRegressor(criterion=self.criterion, max_depth=self.max_depth)
            idx = np.random.choice(range(n_features), size=m, replace=True)
            X_sub = X[idx]
            featureidx.append(idx)
            clf.fit(X_sub, y)
            trees.append(clf)
        self.trees = trees
        self.featureidx = featureidx
        self.X = X
        self.y = y
        return self

    def predict(self, X):
        """
        Funtion to run the RandomForestRegressor on a data point
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_pred=[]
        for i in range(len(X)):
            X_test = X[X.index==i]
            preds=[]
            for j in range(self.n_estimators):
                X_tt = X_test[self.featureidx[j]]
                pred = self.trees[j].predict(X_tt)
                preds.append(pred[0])
            y_pred.append(mean(preds))

        return pd.Series(y_pred)

    def plot(self):
        """
        Function to plot for the RandomForestClassifier.
        It creates three figures

        1. Creates a figure with 1 row and `n_estimators` columns. Each column plots the learnt tree. If using your sklearn, this could a matplotlib figure.
        If using your own implementation, it could simply invole print functionality of the DecisionTree you implemented in assignment 1 and need not be a figure.

        2. Creates a figure showing the decision surface/estimation for each estimator. Similar to slide 9, lecture 4

        3. Creates a figure showing the combined decision surface/prediction

        """
        
        h = .02
        i=1
        fig1 = plt.figure(figsize=(27, 9))
        subXs = self.subXs
        X = self.X
        y = self.y
        x_min, x_max = X[X.columns[0]].min() - .5, X[X.columns[0]].max() + .5
        y_min, y_max = X[X.columns[1]].min() - .5, X[X.columns[1]].max() + .5
        # z_min, z_max = X[X.columns[2]].min() - .5, X[X.columns[2]].max() + .5
        # a_min, a_max = X[X.columns[3]].min() - .5, X[X.columns[3]].max() + .5

        # xx, yy, zz, aa = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h), np.arange(a_min, a_max, h))
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
       
        for treeo in self.trees:
            ax = plt.subplot(1, len(self.trees) , i)
            X = pd.DataFrame(subXs[i-1])
            y = self.y
            # x_min, x_max = X[X.columns[0]].min() - .5, X[X.columns[0]].max() + .5
            # y_min, y_max = X[X.columns[1]].min() - .5, X[X.columns[1]].max() + .5
            # xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
            Z = np.array(treeo.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)))
            # print(Z[12])
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            ax.scatter(X[X.columns[0]], X[X.columns[1]], c=y, cmap=cm_bright, edgecolors='k')
            # size=[1000*w for w in self.weights[i-1]]
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel(str(X.columns[0]))
            ax.set_ylabel(str(X.columns[1]))
            plt.title("Estimator "+str(i))
            i+=1
            
        fig2 = plt.figure(figsize=(9,9))
        X = self.X
        y = self.y
        h=1
        ax2 = plt.subplot(1,1,1)
        z_min, z_max = X[X.columns[2]].min() - .5, X[X.columns[2]].max() + .5
        a_min, a_max = X[X.columns[3]].min() - .5, X[X.columns[3]].max() + .5
        xx, yy, zz, aa = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h), np.arange(z_min, z_max, h), np.arange(a_min, a_max, h))

        Z = np.array(self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel(), zz.ravel(), aa.ravel()], columns=X.columns)))
        Z = Z.reshape(xx.shape)
        pca = PCA(n_components=2)
        Z = pca.fit_transform(Z)
        ax2.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        X = pca.fit_transform(X)
        # size=[1000*w for w in self.weights[i-2]]
        ax2.scatter(X[X.columns[0]],X[X.columns[1]], c=y, cmap=cm_bright, edgecolors='k')
        ax2.set_xlim(xx.min(), xx.max())
        ax2.set_ylim(yy.min(), yy.max())
        plt.title("Combined Decision Surface")
        
        plt.tight_layout()
        plt.show()

        return [fig1,fig2]