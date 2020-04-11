#Libraries you might need 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
#Data Aqusition
dataset = pd.read_csv('C:/Users/Cipher/Desktop/6.1 Machine_Learning_A-Z_New.zip/Machine Learning A-Z New/Part 1 - Data Preprocessing/Part 1 - Data Preprocessing/Data.csv')
#Dependent or feature Variables
features = dataset.iloc[:,:-1].values
#Independent or labels Variables
labels = dataset.iloc[:,3].values
#Missing data (NaN) handling
imputer = SimpleImputer(missing_values=np.nan,strategy="mean")
features[:,1:3] = imputer.fit_transform(features[:,1:3])
#Handling Categorical Data -> Numerical data conversions
LabelEncoder_features = LabelEncoder()
features[:,0] = LabelEncoder_features.fit_transform(features[:,0])
columnTransformer = ColumnTransformer([('encoder', OneHotEncoder(), [0])],remainder='passthrough')
features=np.array(columnTransformer.fit_transform(features))
LabelEncoder_labels = LabelEncoder()
labels = LabelEncoder_features.fit_transform(labels)
#Spliting dataset into training data and testing data
X_train,X_test,y_trian,y_test= train_test_split(features,labels,test_size =0.2,random_state=0)
#Scaling (data might be diverse now normalizing the data)
SC_features = StandardScaler()
X_train = SC_features.fit_transform(X_train)
X_test = SC_features.transform(X_test)