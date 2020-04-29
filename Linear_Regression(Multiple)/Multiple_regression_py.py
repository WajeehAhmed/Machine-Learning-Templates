# -*- coding: utf-8 -*-
"""
Created on Sun Apr 12 10:44:23 2020

@author: Cipher
"""
#Libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import statsmodels.api as smf

#Data Aqusition
dataset = pd.read_csv('50_Startups.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,4].values

#Encoding Categoriacal Data
LabelEncoder_features = LabelEncoder()
X[:,3] = LabelEncoder_features.fit_transform(X[:,3])
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [3])],remainder='passthrough')
X=np.array(columnTransformer.fit_transform(X))

#Avoiding dumy trap
X = X[:,1:]

#Train/Test split   
X_train,X_test,y_trian,y_test= train_test_split(X,y,test_size =0.2,random_state=0)

#Linear Regression
regressor = LinearRegression()
regressor.fit(X_train,y_trian)

#predict
y_pred = regressor.predict(X_test)

#Build optimal model using Backward Elimination
X = np.append(arr= np.ones((50,1)).astype(int),values= X,axis = 1)

X_opt = np.array(X[:, [0, 1, 2, 3, 4, 5]], dtype=float)
regressor_ols = smf.OLS(endog = y,exog = X_opt).fit()

regressor_ols.summary()

X_opt = np.array(X[:, [0, 1, 3, 4, 5]], dtype=float)
regressor_ols = smf.OLS(endog = y,exog = X_opt).fit()

regressor_ols.summary()

X_opt = np.array(X[:, [0, 3, 4, 5]], dtype=float)
regressor_ols = smf.OLS(endog = y,exog = X_opt).fit()

regressor_ols.summary()

X_opt = np.array(X[:, [0, 3, 5]], dtype=float)
regressor_ols = smf.OLS(endog = y,exog = X_opt).fit()

regressor_ols.summary()

X_opt = np.array(X[:, [0, 3]], dtype=float)
regressor_ols = smf.OLS(endog = y,exog = X_opt).fit()

regressor_ols.summary()

#Final optimal X is X_opt