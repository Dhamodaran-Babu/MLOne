from sklearn.feature_selection import chi2
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


def chi_test(dataframe):
    print("\n\n<<<<SELECTING SIGNIFICANT FEATURES>>>>\n\n")
    scaler = MinMaxScaler(feature_range=(0,10))
    x = scaler.fit_transform(dataframe.iloc[:,:-1])
    chi_score , p_val = chi2(x,dataframe.iloc[:,-1])
    p_val = np.round(p_val,5)
    feature_ind = np.where(p_val<0.1)
    print(pd.DataFrame(np.concatenate([chi_score.reshape(-1,1),p_val.reshape(-1,1)],axis=1),
                        index=dataframe.columns[:-1],columns=['chi_score','p_val']))
    seleceted_features = list(dataframe.columns[feature_ind])
    print("selected Features : ",seleceted_features)
    return seleceted_features