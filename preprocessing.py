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
    if(unique_values/len(target)>0.75):
        print("Regression")
        return "Regression"
    else:
        print("Classification")
        return("Classification")

        
def missing_values(data):
    df=pd.DataFrame(data)
    cols = df.columns
    num_cols = list(df._get_numeric_data().columns)
    #print(num_cols)
    char_cols= list(set(cols) - set(num_cols))
    #print(char_cols)
    #print(df.head())
    num_filled=df[num_cols].fillna(df[num_cols].mean())
    #print(num_filled.head())
    char_filled=df[char_cols].fillna(df[char_cols].mode().iloc[0])
    #print(char_filled.head())
    imputed_data=pd.concat([char_filled, num_filled],axis=1)
    #print(imputed_data.head())
    return imputed_data,char_cols
    
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
    
    
    
    
    
    
data=pd.read_csv("hcvdat0.csv",index_col=0)
#print(data.head())
model = targetcheck(data)
data,char_cols=missing_values(data)
"""if(model == "Classification"):
    classification_preprocessing(data,char_cols)
else:
    """