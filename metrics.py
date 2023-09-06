import numpy as np
import pandas as pd

def accuracy(y_hat, y):
    assert(y_hat.size == y.size)
    # TODO: Write here
    count = 0
    for i in range(y_hat.size):
        if(y_hat.iat[i]==y.iat[i]):
            count+=1
    
    return count/y.size


def precision(y_hat, y, cls):
    assert(y_hat.size==y.size)

    TruePositives = 0
    TP_plus_FP = 0
    for i in range(y_hat.size):
        if(y_hat.iat[i]==cls):
            if(y_hat.iat[i]==y.iat[i]):
                TruePositives+=1
            TP_plus_FP+=1
    
    if(TruePositives==0):
        return 0
    
    return TruePositives/TP_plus_FP


def recall(y_hat, y, cls):
    assert(y_hat.size==y.size)

    TruePositives = 0
    TP_plus_TN = 0
    for i in range(y_hat.size):
        if(y.iat[i]==cls):
            if(y_hat.iat[i]==y.iat[i]):
                TruePositives+=1
            TP_plus_TN+=1
    
    if(TruePositives==0):
        return 0
    
    return TruePositives/TP_plus_TN


def rmse(y_hat, y):
    assert(y.size==y_hat.size)

    diffSqSum = 0
    for i in range(y.size):
        diffSqSum += (y_hat.iloc[i]-y.iloc[i])**2
    
    result = np.sqrt(diffSqSum/y.size)
    return result

    

def mae(y_hat, y):
    assert(y_hat.size==y.size)

    diffAbsSum = 0
    for i in range(y.size):
        diffAbsSum += abs(y_hat.iloc[i]-y.iloc[i])
    
    result = (diffAbsSum/y.size)
    
    return result