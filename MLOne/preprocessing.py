# -*- coding: utf-8 -*-
"""
Created on Fri Oct  2 12:28:56 2020

@author: Sri Suhas
"""
import pandas as pd
import numpy as np

def targetcheck(data):
    """ Checks if the dataset is of classification model or regression model"""
    target = data.iloc[:,-1]
    unique_values = target.nunique()
    if(unique_values/len(target)>0.65):
        return "Regression"
    else:
        return("Classification")

def missing_values(df):
    cols = df.columns
    num_cols = list(df._get_numeric_data().columns)
    char_cols= list(set(cols) - set(num_cols))
    for cols in char_cols:
        df[cols] = df[cols].fillna(value=df[cols].mode()[0])
    for cols in num_cols:
        df[cols] = df[cols].fillna(value=df[cols].mean())
    return df,char_cols

def remove_outliers(df,cat_cols):
    num_cols = list(set(df.columns) - set(cat_cols))
    for col in num_cols:
        if col is not df.columns[-1]:
            feature = df[col]
            sorted(feature)
            q1,q3 = feature.quantile([0.25,0.75])
            iqr = q3-q1
            upper_limit = q3 + (1.5 * iqr)
            lower_limit = q1 - (1.5 * iqr)
            df[col] = np.where(df[col]>upper_limit,upper_limit,df[col])
            df[col] = np.where(df[col]<lower_limit,lower_limit,df[col])
    return df

def balance_the_data(df):
    "Balance the data with SMOTE over sampling"

    from imblearn.over_sampling import SMOTE
    over_sampler = SMOTE(sampling_strategy='all',random_state=101,n_jobs=-1)
    x,y = over_sampler.fit_resample(df.iloc[:,:-1],df.iloc[:,-1])
    return x,y

def classification_preprocessing(df,cat_cols):

    "Encode the categorical values"
    from sklearn.preprocessing import LabelEncoder
    encoder = LabelEncoder()
    for col in cat_cols:
        df[col] = encoder.fit_transform(df[col])

    "Select significant varibles with chi-square test"
    from MLOne.select_variables import chi_test
    selected_features = chi_test(df)
    feature_selected_df = df[selected_features+[df.columns[-1]]]

    "Remove imbalance in the data"
    x,y = balance_the_data(feature_selected_df)
    
    "Standard Scaling the data"
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    x = scaler.fit_transform(x)
    return x,y
    
def regresssion_preprocessing(data):
    from sklearn.preprocessing import LabelEncoder,OneHotEncoder
    x=data.iloc[:,:-1]
    y=data.iloc[:,-1]
    onehot_encoder = OneHotEncoder(sparse=False)
    X = onehot_encoder.fit_transform(x)
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(y)
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X,Y
    
def preprocessor (data):
    print("\n\n<<<<PREPROCESSING THE DATA>>>>\n\n")
    pbm_type = targetcheck(data)
    df,cat_cols = missing_values(data)
    df = remove_outliers(df,cat_cols = cat_cols)
    if pbm_type is "Classification":
        x,y = classification_preprocessing(df,cat_cols=cat_cols)
    else :
        x,y = regresssion_preprocessing(df)

    return x,y,pbm_type