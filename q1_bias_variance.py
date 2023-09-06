
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor 
from sklearn.model_selection import train_test_split

np.random.seed(42)
x = np.linspace(0, 10, 50)
eps = np.random.normal(0, 5, 50)
y = x**2 + 1 + eps



bias=[]
variance=[]

#for i depth
for i in range(1,7):
    temp_bias = []
    temp_variance = []
    for j in range(100):
        mask = np.random.choice(50, size=50)
        sample_x=x[mask]
        sample_y=y[mask]
        tree= DecisionTreeRegressor(max_depth=i)
        X_train, X_test, y_train, y_test = train_test_split(sample_x, sample_y, test_size=0.3)
        sample_x = pd.DataFrame(X_train)
        sample_y = pd.Series(y_train)
        X_test = pd.DataFrame(X_test)
        y_test = pd.Series(y_test)
        tree.fit(sample_x,sample_y)
        y_hat=tree.predict(X_test)
        bias_value=np.mean(abs((y_hat)-(y_test)))**2
        temp_bias.append(bias_value)
        variance_value=np.var(y_hat)
        temp_variance.append(variance_value)
    bias.append(np.mean(temp_bias))
    variance.append(np.mean(temp_variance))
    

#normalize
variance=(variance-min(variance))/(max(variance)-min(variance))
bias=(bias-min(bias))/(max(bias)-min(bias))

depth=np.arange(1,7)
plt.plot(depth,bias,'r-')
plt.plot(depth,variance,'g-')
plt.xlabel('Max Depth')
plt.savefig("./Results/BiasVsVarTest.png")
plt.clf()





#####################################################
np.random.seed(42)
x = np.linspace(0, 10, 50)
eps = np.random.normal(0, 5, 50)
y = x**2 + 1 + eps



bias=[]
variance=[]


for i in range(1,7):
    temp_bias = []
    temp_variance = []
    for j in range(100):
        mask = np.random.choice(50, size=50)
        sample_x=x[mask]
        sample_y=y[mask]
        tree= DecisionTreeRegressor(max_depth=i)

        sample_x = pd.DataFrame(sample_x)
        sample_y = pd.Series(sample_y)
        tree.fit(sample_x,sample_y)
        y_hat=tree.predict(sample_x)
        bias_value=np.mean(abs((y_hat)-(sample_y)))**2
        temp_bias.append(bias_value)
        variance_value=np.var(y_hat)
        temp_variance.append(variance_value)
    bias.append(np.mean(temp_bias))
    variance.append(np.mean(temp_variance))
    


variance=(variance-min(variance))/(max(variance)-min(variance))
bias=(bias-min(bias))/(max(bias)-min(bias))

depth=np.arange(1,7)
plt.plot(depth,bias,'r-')
plt.plot(depth,variance,'g-')
plt.xlabel('Max Depth')
plt.savefig("./Results/BiasVsVar.png")