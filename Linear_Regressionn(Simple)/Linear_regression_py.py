# -*- coding: utf-8 -*-
"""
Created on Sat Apr 11 06:41:15 2020

@author: Cipher
"""
#Libraries you might need
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LinearRegression

#data aqusition
dataset = pd.read_csv('Salary_Data.csv')
X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

#spliting dataset into training and testing data
X_train,X_test,y_trian,y_test= train_test_split(X,y,test_size =1/3,random_state=0)

#Regressor
regressor = LinearRegression()

regressor.fit(X_train,y_trian)

#prediction vector
y_pred = regressor.predict(X_test)

#visualization
plt.scatter(X_train,y_trian,color = 'blue')
plt.plot(X_train,regressor.predict(X_train),color =  'red')
plt.title("Salary VS Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()
#visualization with testing data
plt.scatter(X_test,y_test,color = 'blue')
plt.plot(X_train,regressor.predict(X_train),color =  'red')
plt.title("Salary VS Experience")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

