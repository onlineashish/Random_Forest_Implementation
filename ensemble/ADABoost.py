
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import copy

class AdaBoostClassifier():
    def __init__(self, base_estimator, n_estimators=3): # Optional Arguments: Type of estimator
        '''
        :param base_estimator: The base estimator model instance from which the boosted ensemble is built (e.g., DecisionTree, LinearRegression).
                               If None, then the base estimator is DecisionTreeClassifier(max_depth=1).
                               You can pass the object of the estimator class
        :param n_estimators: The maximum number of estimators at which boosting is terminated. In case of perfect fit, the learning procedure may be stopped early.
        '''
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators

        

    def fit(self, X, y):
        """
        Function to train and construct the AdaBoostClassifier
        Inputs:
        X: pd.DataFrame with rows as samples and columns as features (shape of X is N X P) where N is the number of samples and P is the number of columns.
        y: pd.Series with rows corresponding to output variable (shape of Y is N)
        """
        classifiers = []
        estimator_alphas = []
        w = [1/len(X) for i in range(len(X))]
        self.weights = []
        for i in range(self.n_estimators): 
            self.weights.append(copy.deepcopy(w))        
            clf = self.base_estimator.fit(X,y, w)
            y_pred = clf.predict(X)
            incorrect = []
            for i in range(len(y)):
                if y[i] != y_pred[i]:
                    incorrect.append(i)
            if(len(incorrect) == 0):
                self.n_estimators = i+1
                break
            totalerror = 0
            for ind in incorrect:
                totalerror += w[ind]
            totalerror = totalerror/sum(w)
            significance = (1/2)*np.log((1-totalerror)/totalerror) #alpha = 1/2 log((1-err)/err)
            for i in range(len(w)):
                if i in incorrect:
                    w[i] = w[i]*np.exp(significance)            #w_updated = prev_weight * e^alpha
                else:
                    w[i] = w[i]*np.exp(-significance)
            estimator_alphas.append(significance)
            norm_w = [float(i)/sum(w) for i in w]
            w = norm_w
            classifiers.append(copy.deepcopy(clf))

        self.classifiers = classifiers
        self.estimator_alphas = estimator_alphas
        self.classes = np.unique(y)
        self.X = X
        self.y = y

        

    def predict(self, X):
        """
        Input:
        X: pd.DataFrame with rows as samples and columns as features
        Output:
        y: pd.Series with rows corresponding to output variable. THe output variable in a row is the prediction for sample in corresponding row in X.
        """
        y_pred = []
        for j in X.index.values:              
            pred_scores = [0 for i in self.classes]
            for i in range(self.n_estimators):
                clf = self.classifiers[i]
                alpha = self.estimator_alphas[i]
                pred = clf.predict(X[X.index==j])
                pred_scores[pred[0]] += alpha
            final_pred = np.argmax(pred_scores)
            y_pred.append(final_pred)
        
        return pd.Series(y_pred)


    def plot(self):
        """
        Function to plot the decision surface for AdaBoostClassifier for each estimator(iteration).
        Creates two figures
        Figure 1 consists of 1 row and `n_estimators` columns
        The title of each of the estimator should be associated alpha (similar to slide#38 of course lecture on ensemble learning)
        Further, the scatter plot should have the marker size corresponnding to the weight of each point.

        Figure 2 should also create a decision surface by combining the individual estimators

        Reference for decision surface: https://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html

        This function should return [fig1, fig2]
        """
        h = .02
        
        X = self.X
        y = self.y
        fig1 = plt.figure(figsize=(27, 9))
        x_min, x_max = X[X.columns[0]].min() - .5, X[X.columns[0]].max() + .5
        y_min, y_max = X[X.columns[1]].min() - .5, X[X.columns[1]].max() + .5
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        cm = plt.cm.RdBu
        cm_bright = ListedColormap(['#FF0000', '#0000FF'])
        
        i=1
        for clf in self.classifiers:
            ax = plt.subplot(1, len(self.classifiers) , i)
            Z = np.array(clf.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)))
            
            Z = Z.reshape(xx.shape)
            ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)
            size=[1000*w for w in self.weights[i-1]]
            ax.scatter(X[X.columns[0]], X[X.columns[1]], c=y, s=size, cmap=cm_bright, edgecolors='k')
            ax.set_xlim(xx.min(), xx.max())
            ax.set_ylim(yy.min(), yy.max())
            ax.set_xlabel(str(X.columns[0]))
            ax.set_ylabel(str(X.columns[1]))
            plt.title("Classifier "+str(i)+": Alpha="+str(self.estimator_alphas[i-1]))
            i+=1
            
        fig2 = plt.figure(figsize=(9,9))
        ax2 = plt.subplot(1,1,1)
        Z = np.array(self.predict(pd.DataFrame(np.c_[xx.ravel(), yy.ravel()], columns=X.columns)))
        Z = Z.reshape(xx.shape)
        ax2.contourf(xx, yy, Z, cmap=cm, alpha=.8)
        size=[1000*w for w in self.weights[i-2]]
        ax2.scatter(X[X.columns[0]], X[X.columns[1]], c=y, s=size, cmap=cm_bright, edgecolors='k')
        ax2.set_xlim(xx.min(), xx.max())
        ax2.set_ylim(yy.min(), yy.max())
        plt.title("Combined Decision Surface")
        
        plt.tight_layout()
        plt.show()

        return [fig1,fig2]
