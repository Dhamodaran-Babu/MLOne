import numpy as np
import pandas as pd

def uniform_splitter(x,y):
    from sklearn.model_selection import StratifiedShuffleSplit
    splitter = StratifiedShuffleSplit(n_splits=1,test_size=0.1,random_state=101)
    for train_ind,val_ind in splitter.split(x,y):
        xtrain = x[train_ind];xval = x[val_ind]
        ytrain = y[train_ind];yval = y[val_ind]
    return xtrain,xval,ytrain,yval

def train_test_validation_splitter(x,y,pbm_type):
    print("\n\n<<<<PERFORMING TRAIN TEST AND VALIDATION SPLIT>>>>\n\n")
    x = np.array(x);y=np.array(y).reshape(-1,1)
    if len(x.shape)==1 :
        x = x.reshape(-1,1)
    
    if pbm_type is "Classification":
        xtrain,xval,ytrain,yval = uniform_splitter(x,y)

    elif pbm_type is "Regression":
        from sklearn.model_selection import train_test_split
        xtrain,xval,ytrain,yval = train_test_split(x.values,y.values,test_size=0.1,random_state=101)
    
    return xtrain,xval,ytrain,yval