"""
The current code given is for the Assignment 1.
You will be expected to use this to make trees for:
> discrete input, discrete output
> real input, real output
> real input, discrete output
> discrete input, real output
"""
import copy
import numpy as np
import pandas as pd
import operator
import matplotlib.pyplot as plt
from .utils import entropy, information_gain, gini_index

np.random.seed(42)
# creating class for a tree node
class TreeNode():
    def __init__(self):
        self.isLeaf = False
        self.isAttrCategory = False
        self.attr_id = None
        self.children = dict()
        self.splitValue = None
        self.value = None



class DecisionTree():
    def __init__(self, criterion, max_depth=None):
        self.criterion = criterion
        self.max_depth = max_depth
        self.head = None

    
    def fit_data(self,X,y,currdepth,weights=None):

        currnode = TreeNode()   # Creating a new Tree Node

        attr_id = -1
        split_value = None
        best_measure = None

        # Classification Problems
        if(y.dtype.name=="category"):
            classes = np.unique(y)
            if(classes.size==1):
                currnode.isLeaf = True
                currnode.isAttrCategory = True
                currnode.value = classes[0]
                return currnode
            if(self.max_depth!=None):
                if(self.max_depth==currdepth):
                    currnode.isLeaf = True
                    currnode.isAttrCategory = True
                    classes = dict()
                    for i in range(y.size):
                        if(y.iat[i] in classes):
                            classes[y.iat[i]] += weights.iat[i]
                        else:
                            classes[y.iat[i]] = weights.iat[i]
                    currnode.value = max(classes.items(), key=operator.itemgetter(1))[0]
                    return currnode
            if(X.shape[1]==0):
                currnode.isLeaf = True
                currnode.isAttrCategory = True
                for i in range(y.size):
                    if(y.iat[i] in classes):
                        classes[y.iat[i]] += weights.iat[i]
                    else:
                        classes[y.iat[i]] = weights.iat[i]
                currnode.value = max(classes.items(), key=operator.itemgetter(1))[0]
                return currnode


            for i in X:
                xcol = X[i]
                
                # Discreate Input and Discreate Output
                if(xcol.dtype.name=="category"):
                    measure = None
                    if(self.criterion=="information_gain"):         # Criteria is Information Gain
                        measure = information_gain(y,xcol,weights)
                    else:                                           # Criteria is Gini Index
                        classes1 = np.unique(xcol)
                        s = 0
                        for j in classes1:
                            y_sub = pd.Series([y[k] for k in range(y.size) if xcol[k]==j])
                            s += y_sub.size*gini_index(y_sub,weights)
                        measure = -1*(s/xcol.size)
                    if(best_measure!=None):
                        if(best_measure<measure):
                            attr_id = i
                            best_measure = measure
                            split_value = None
                    else:
                        attr_id = i
                        best_measure = measure
                        split_value = None
                
                # Real Input and Discreate Output
                else:
                    xcol_sorted = xcol.sort_values()
                    for j in range(xcol_sorted.size-1):
                        index = xcol_sorted.index[j]
                        next_index = xcol_sorted.index[j+1]
                        if(y[index]!=y[next_index]):
                            measure = None
                            splitValue = (xcol[index]+xcol[next_index])/2
                            
                            if(self.criterion=="information_gain"):                 # Criteria is Information Gain
                                helper_attr = pd.Series(xcol<=splitValue)
                                measure = information_gain(y,helper_attr,weights)
                            
                            else:                                                   # Criteria is Gini Index
                                y_sub1 = pd.Series([y[k] for k in range(y.size) if xcol[k]<=splitValue])
                                y_sub2 = pd.Series([y[k] for k in range(y.size) if xcol[k]>splitValue])
                                measure = y_sub1.size*gini_index(y_sub1,weights) + y_sub2.size*gini_index(y_sub2,weights)
                                measure =  -1*(measure/np.sum(weights))
                            if(best_measure!=None):
                                if(best_measure<measure):
                                    attr_id = i
                                    best_measure = measure
                                    split_value = splitValue
                            else:
                                attr_id = i
                                best_measure = measure
                                split_value = splitValue
            
        
        # Regression Problems
        else:
            if(self.max_depth!=None):
                if(self.max_depth==currdepth):
                    currnode.isLeaf = True
                    currnode.value = y.mean()
                    return currnode
            if(y.size==1):
                currnode.isLeaf = True
                currnode.value = y[0]
                return currnode
            if(X.shape[1]==0):
                currnode.isLeaf = True
                currnode.value = y.mean()
                return currnode
            

            for i in X:
                xcol = X[i]

                # Discreate Input Real Output
                if(xcol.dtype.name=="category"):
                    classes1 = np.unique(xcol)
                    measure = 0
                    for j in classes1:
                        y_sub = pd.Series([y[k] for k in range(y.size) if xcol[k]==j])
                        measure += y_sub.size*np.var(y_sub)
                    if(best_measure!=None):
                        if(best_measure>measure):
                            best_measure = measure
                            attr_id = i
                            split_value = None
                    else:
                        best_measure = measure
                        attr_id = i
                        split_value = None
                
                # Real Input Real Output
                else:
                    xcol_sorted = xcol.sort_values()
                    for j in range(y.size-1):
                        index = xcol_sorted.index[j]
                        next_index = xcol_sorted.index[j+1]
                        splitValue = (xcol[index]+xcol[next_index])/2
                        y_sub1 = pd.Series([y[k] for k in range(y.size) if xcol[k]<=splitValue])
                        y_sub2 = pd.Series([y[k] for k in range(y.size) if xcol[k]>splitValue])
                        measure = y_sub1.size*np.var(y_sub1) + y_sub2.size*np.var(y_sub2)
                        # c1 = y_sub1.mean()
                        # c2 = y_sub2.mean()
                        # measure = np.mean(np.square(y_sub1-c1) + np.square(y_sub2-c2))
                        if(best_measure!=None):
                            if(best_measure>measure):
                                attr_id = i
                                best_measure = measure
                                split_value = splitValue
                        else:
                            attr_id = i
                            best_measure = measure
                            split_value = splitValue
        

        # when current treenode is category based
        if(split_value==None):
            currnode.isAttrCategory = True
            currnode.attr_id = attr_id
            classes = np.unique(X[attr_id])
            for j in classes:
                y_new = pd.Series([y[k] for k in range(y.size) if X[attr_id][k]==j], dtype=y.dtype)
                X_new = X[X[attr_id]==j].reset_index().drop(['index',attr_id],axis=1)
                new_weights = pd.Series([weights[k] for k in range(y.size) if X[attr_id][k]==j])
                currnode.children[j] = self.fit_data(X_new, y_new, currdepth+1, new_weights)
        # when current treenode is split based
        else:
            currnode.attr_id = attr_id
            currnode.splitValue = split_value
            y_new1 = pd.Series([y[k] for k in range(y.size) if X[attr_id][k]<=split_value], dtype=y.dtype).reset_index(drop=True)
            X_new1 = X[X[attr_id]<=split_value].reset_index().drop(['index'],axis=1)
            new_weights1 = pd.Series([weights[k] for k in range(y.size) if X[attr_id][k]<=split_value]).reset_index(drop=True)
            y_new2 = pd.Series([y[k] for k in range(y.size) if X[attr_id][k]>split_value], dtype=y.dtype).reset_index(drop=True)
            X_new2 = X[X[attr_id]>split_value].reset_index().drop(['index'],axis=1)
            new_weights2 = pd.Series([weights[k] for k in range(y.size) if X[attr_id][k]>split_value]).reset_index(drop=True)
            if(y_new1.size==0 or y_new2.size==0):
                currnode.isLeaf = True
                if(y.dtype.name=="category"):
                    for i in range(y.size):
                        if(y.iat[i] in classes):
                            classes[y.iat[i]] += weights.iat[i]
                        else:
                            classes[y.iat[i]] = weights.iat[i]
                    currnode.value = max(classes.items(), key=operator.itemgetter(1))[0]
                else:
                    currnode.value = y.mean()
                return currnode
            currnode.children["lessThan"] = self.fit_data(X_new1, y_new1, currdepth+1, new_weights1)
            currnode.children["greaterThan"] = self.fit_data(X_new2, y_new2, currdepth+1, new_weights2)
        
        return currnode


    def fit(self, X, y, weights=None):
        assert(y.size>0)
        assert(X.shape[0]==y.size)
        self.head = self.fit_data(X,y,0, weights)
        return copy.deepcopy(self)


    def predict(self, X):
        y_hat = list()                  # List to contain the predicted values

        for i in range(X.shape[0]):
            xrow = X.iloc[i,:]          # Get an instance of the data for prediction purpose

            h = self.head
            while(not h.isLeaf):                            # when treenode is not a leaf
                if(h.isAttrCategory):                       # when treenode is category based
                    h = h.children[xrow[h.attr_id]]
                else:                                       # when treenode is split based
                    if(xrow[h.attr_id]<=h.splitValue):
                        h = h.children["lessThan"]
                    else:
                        h = h.children["greaterThan"]
            
            y_hat.append(h.value)                           #when treenode is a leaf
        
        y_hat = pd.Series(y_hat)

        return y_hat


    def plotTree(self, root, depth):
        if(root.isLeaf):
            if(root.isAttrCategory):
                return "Class "+str(root.value)
            else:
                return "Value "+str(root.value)

        s = ""
        if(root.isAttrCategory):
            for i in root.children.keys():
                s += "?("+str(root.attr_id)+" == "+str(i)+")\n" 
                s += "\t"*(depth+1)
                s += str(self.plotTree(root.children[i], depth+1)).rstrip("\n") + "\n"
                s += "\t"*(depth)
            s = s.rstrip("\t")
        else:
            s += "?("+str(root.attr_id)+" <= "+str(root.splitValue)+")\n"
            s += "\t"*(depth+1)
            s += "Y: " + str(self.plotTree(root.children["lessThan"], depth+1)).rstrip("\n") + "\n"
            s += "\t"*(depth+1)
            s += "N: " + str(self.plotTree(root.children["greaterThan"], depth+1)).rstrip("\n") + "\n"
        
        return s
           

    def plot(self):
        h = self.head
        s = self.plotTree(h,0)
        print(s)