from tree.base import DecisionTree
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from metrics import *


#DataSet
from sklearn.datasets import make_classification
X, y = make_classification(
n_features=2, n_redundant=0, n_informative=2, random_state=1, n_clusters_per_class=2, class_sep=0.5)

X = pd.DataFrame(X)
y = pd.Series(y,dtype="category")

split_percent = int(0.7*len(y))


X_train = X.iloc[:split_percent, :]
X_train = pd.DataFrame(X_train)

X_test = X.iloc[split_percent:, :]
X_test = pd.DataFrame(X_test)

y_train = y.iloc[:split_percent]
y_train = pd.Series(y_train,dtype="category")

y_test = y.iloc[split_percent:]
y_test = pd.Series(y_test,dtype="category")

# Generate random weights for the training set
weights = np.random.uniform(0, 1, size=X_train.shape[0])
weights = pd.Series(weights)

######
# Learning the classifier and testing 
criteria = 'information_gain'
tree = DecisionTree(criterion=criteria) 
tree.fit(X_train, y_train, weights=weights)
y_hat = tree.predict(X_test)
# tree.plot()
print('Manual implementation')
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('class:',cls)
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))


###########Sklearn- classifier###############

print()
print()
print()

###########Sklearn- classifier###############


criteria = 'entropy'
sk_tree = DecisionTreeClassifier(criterion=criteria) 
sk_tree.fit(X_train, y_train, sample_weight=weights)
y_hat = sk_tree.predict(X_test)
y_hat = pd.Series(y_hat,dtype="category")
# tree.plot()
print('Sklearn data')
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y_test))
for cls in y.unique():
    print('class:',cls)
    print('Precision: ', precision(y_hat, y_test, cls))
    print('Recall: ', recall(y_hat, y_test, cls))







#######################################
'''
            PLOT

'''
#######################################

def plot(model,X,y,figname):

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(4, 4))


    title=f"Decision surface/ boundary"
    #self.fig1.append(temp)
    X_arr=X.values
    y_arr=y.values
    x1_min, x1_max = X_arr[:, 0].min() - 0.2, X_arr[:, 0].max() + 0.2
    x2_min, x2_max = X_arr[:, 1].min() - 0.2, X_arr[:, 1].max() + 0.2
    xx, yy = np.meshgrid(np.linspace(x1_min, x1_max, 100),
                         np.linspace(x2_min, x2_max, 100))
    test=np.c_[xx.ravel(), yy.ravel()]
    test = pd.DataFrame(test)
    Z = model.predict(test)
    
    if model == tree:
        Z = Z.values
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4)
    ax.scatter(X_arr[:, 0], X_arr[:, 1], c=y_arr, edgecolor='k')
    ax.set_title(title)

        #plot_decision_boundary(ax,f"alpha_{i} ={self.alphas[i]}",self.my_models[i], X_arr, y_arr, self.model_wts[i])
    plt.savefig(figname)
location_sk = "./Results/Q2_sk.png"
location_manual = "./Results/Q2_manual.png"
plot(sk_tree, X, y, location_sk)
plot(tree, X, y, location_manual)
