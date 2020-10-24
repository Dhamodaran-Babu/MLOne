from sklearn.feature_selection import chi2
import pandas as pd
import numpy as np


def chi_test(dataframe):
    chi_score , p_val = chi2(dataframe.iloc[:,:-1],dataframe.iloc[:,-1])
    p_val = np.round(p_val,5)
    feature_ind = np.where(p_val<0.1)
    print(pd.DataFrame(np.concatenate([chi_score.reshape(-1,1),p_val.reshape(-1,1)],axis=1),
                        index=dataframe.columns[:-1],columns=['chi_score','p_val']))
    seleceted_features = list(dataframe.columns[feature_ind])
    print("selected Features : ",seleceted_features)
    return seleceted_features