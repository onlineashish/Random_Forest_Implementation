import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from metrics import *

from ensemble.ADABoost import AdaBoostClassifier
from tree.base import DecisionTree

# Or you could import sklearn DecisionTree

np.random.seed(42)

########### AdaBoostClassifier on Real Input and Discrete Output ###################


N = 30
P = 2
NUM_OP_CLASSES = 2
n_estimators = 4
X = pd.DataFrame(np.abs(np.random.randn(N, P)))
y = pd.Series(np.random.randint(NUM_OP_CLASSES, size = N), dtype="category")

criteria = 'entropy'
tree = DecisionTreeClassifier(criterion=criteria,max_depth=1)
Classifier_AB = AdaBoostClassifier(base_estimator=tree, n_estimators=n_estimators )
Classifier_AB.fit(X, y)
y_hat = Classifier_AB.predict(X)
[fig1, fig2] = Classifier_AB.plot()
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print("For class "+str(cls))
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))
loc1 ="./Results/Q3_fig1.png"
loc2 ="./Results/Q3_fig2.png"
fig1.savefig(loc1)
fig2.savefig(loc2)


print()
print()

#############   Decision Stumb     ############
criteria = 'entropy'
Sk_tree = DecisionTreeClassifier(criterion=criteria,max_depth=1)
Sk_tree.fit(X,y)
y_hat = Sk_tree.predict(X)
y_hat = pd.Series(y_hat)
print('Criteria :', criteria)
print('Accuracy: ', accuracy(y_hat, y))
for cls in y.unique():
    print("For class "+str(cls))
    print('Precision: ', precision(y_hat, y, cls))
    print('Recall: ', recall(y_hat, y, cls))


