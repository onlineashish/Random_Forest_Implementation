import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
from metrics import *
from sklearn.tree import DecisionTreeClassifier
from ensemble.bagging import BaggingClassifier
import os
# Or use sklearn decision tree

########### BaggingClassifier ###################


time_list_sp =[]
time_list_mp =[]
for i in range(10000,100001,10000):
    N = i
    P = 2
    NUM_OP_CLASSES = 2
    n_estimators = 3
    np.random.seed(42)
    X = pd.DataFrame(np.abs(np.random.randn(N, P)))
    y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")
    
    criteria = "entropy"
    tree = DecisionTreeClassifier(criterion=criteria)
    Classifier_Ba = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators, n_jobs=1)
    start_time = time.perf_counter()
    Classifier_Ba.fit(X, y)
    end_time = time.perf_counter()
    time_list_sp.append(end_time - start_time)

    y_hat = Classifier_Ba.predict(X)
    #[fig1, fig2] = Classifier_B.plot()
    # fig1.savefig("./Results/Q4_Bagging_fig1_0.png")
    # fig2.savefig("./Results/Q4_Bagging_fig2_0.png")
    # print("Criteria :", criteria)
    # print("Accuracy: ", accuracy(y_hat, y))
    # for cls in y.unique():
    #     print("For class "+str(cls))
    #     print("Precision: ", precision(y_hat, y, cls))
    #     print("Recall: ", recall(y_hat, y, cls))
    #     break


for i in range(10000,100001,10000):

    N = i
    P = 2
    NUM_OP_CLASSES = 2
    n_estimators = 3
    np.random.seed(42)
    X = pd.DataFrame(np.abs(np.random.randn(N, P)))
    y = pd.Series(np.random.randint(NUM_OP_CLASSES, size=N), dtype="category")
    criteria = "entropy"
    tree = DecisionTreeClassifier(criterion=criteria)
    Classifier_B = BaggingClassifier(base_estimator=tree, n_estimators=n_estimators, n_jobs=os.cpu_count())
    start_time = time.perf_counter()
    Classifier_B.fit(X, y)
    end_time = time.perf_counter()
    time_list_mp.append(end_time - start_time)

    y_hat = Classifier_B.predict(X)
    #[fig1, fig2] = Classifier_B.plot()
    # fig1.savefig("./Results/Q4_Bagging_fig1_0.png")
    # fig2.savefig("./Results/Q4_Bagging_fig2_0.png")
    # print("Criteria :", criteria)
    # print("Accuracy: ", accuracy(y_hat, y))
    # for cls in y.unique():
    #     print("For class "+str(cls))
    #     print("Precision: ", precision(y_hat, y, cls))
    #     print("Recall: ", recall(y_hat, y, cls))
    #     break

plt.clf()
y = [i for i in range(10000,100001,10000)]
plt.plot(y,time_list_sp , label = "single processing")
plt.plot(y,time_list_mp , label = "multi processing")
plt.xlabel('size of bagging')
plt.ylabel('time')
plt.title('time comparision')
plt.legend()

# Display the plot
plt.show()
    
plt.savefig("./Results/Q4_timeplot.png")
