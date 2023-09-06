import numpy as np
import pandas as pd


def entropy(Y,weights):
    classes = dict()
    for i in range(Y.size):
        if(Y.iat[i] in classes):
            classes[Y.iat[i]] += weights[i]
        else:
            classes[Y.iat[i]] = weights[i]
    
    entropy = 0
    k = np.sum(weights)
    for i in classes.keys():
        p_i = classes[i]/k
        entropy -= (p_i*np.log2(p_i))
    
    return entropy


def gini_index(Y, weights):
    k = np.sum(weights)
    classes = dict()
    for i in range(Y.size):
        if(Y.iat[i] in classes):
            classes[Y.iat[i]] += weights[i]
        else:
            classes[Y.iat[i]] = weights[i]
    
    gini = 1

    for i in classes.keys():
        p_i = classes[i]/k
        gini -= np.square(p_i)
    
    return gini

    


def information_gain(Y, attr, weights):
    assert(attr.size==Y.size)

    k = np.sum(weights)

    classes_attr = dict()
    for i in range(attr.size):
        if(attr.iat[i] in classes_attr):
            classes_attr[attr.iat[i]][0].append(Y.iat[i])
            classes_attr[attr.iat[i]][1].append(weights.iat[i])
        else:
            classes_attr[attr.iat[i]] = [[Y.iat[i]],[weights.iat[i]]]
    
    gain = entropy(Y, weights)

    for i in classes_attr.keys():
        p_i = sum(classes_attr[i][1])/k
        gain -= (p_i*entropy(pd.Series(data=classes_attr[i][0]).reset_index(drop=True), pd.Series(data=classes_attr[i][1]).reset_index(drop=True)))
    
    return gain
