# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 12:28:56 2020

@author: Sri Suhas
"""
import pandas as pd
import numpy as np

def targetcheck(data):
    """ Checks if the dataset is of classification model or regression model"""
    target = data.iloc[:,1]
    unique_values = target.nunique()
    print(unique_values)
    print(unique_values/len(target))
    if(unique_values/len(target)>0.65):
        print("Regression")
        return "Regression"
    else:
        print("Classification")
        return("Classification")

        
def missing_values(data):
    df=pd.DataFrame(data)
    cols = df.columns
    num_cols = list(df._get_numeric_data().columns)
    char_cols_cols = list(set(cols) - set(num_cols))
    for cols in char_cols:
        df[cols] = df[cols].fillna(value=df[cols].mode()[0])
    for cols in num_cols:
        df[cols] = df[cols].fillna(value=df[cols].mean())
    return df,char_cols
    
def classification_preprocessing(data,char_cols):
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    df=pd.DataFrame(data)
    target = df.iloc[:,0]
    encoder.fit(target)
    #print(encoder.classes_)
    target=encoder.transform(target)
    print(target)
    g = df.groupby(target,group_keys=False)
    balanced_data = pd.DataFrame(g.apply(lambda x: x.sample(g.size().min()))).reset_index(drop=True)
    print(balanced_data)
    from sklearn.preprocessing import StandardScaler
    x = balanced_data.drop(balanced_data[char_cols],axis=1).values
    #y = balanced_data[target].values
    scaler = StandardScaler()
    scaler.fit(x)
    x=scaler.transform(x)
    print(x)        
    
def regresssion_preprocessing(data):
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    x=data.iloc[:,1:]
    y=data.iloc[:,0]
    onehot_encoder = OneHotEncoder(sparse=False)
    X = onehot_encoder.fit_transform(x)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(y)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X,Y
    
    
    
    
data=pd.read_csv("hcvdat0.csv",index_col=0)
#print(data.head())
model = targetcheck(data)
data,char_cols=missing_values(data)
if(model == "Classification"):
    classification_preprocessing(data,char_cols)
else:
    regresssion_preprocessing(data)