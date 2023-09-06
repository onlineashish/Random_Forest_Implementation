import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from metrics import *
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tree.randomForest import RandomForestClassifier


###Write code here


N = 30
P = 5
X = pd.DataFrame(np.random.randn(N, P))
y = pd.Series(np.random.randint(P, size = N), dtype="category")

for criteria in ['entropy', 'gini']:
    Classifier_RF = RandomForestClassifier(5, criterion = criteria, max_depth=5)
    Classifier_RF.fit(X, y)
    y_hat = Classifier_RF.predict(X)
    fig1,fig2 = Classifier_RF.plot()
    fig1.savefig("./Results/Q5_RF_C_fig1.png")
    fig2.savefig("./Results/Q5_RF_C_fig2.png")
    print('Criteria :', criteria)
    print('Accuracy: ', accuracy(y_hat, y))
    for cls in y.unique():
        print("For class "+str(cls))
        print('Precision: ', precision(y_hat, y, cls))
        print('Recall: ', recall(y_hat, y, cls))


