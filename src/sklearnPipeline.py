import pandas as pd 
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler ,RobustScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

#------------------------------------------absolute path----------------


import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

#splitting
#pipeline ==> polyfeatures --> scaling -->model
#-----------------------------------------data----------------------------
data=pd.read_csv(os.path.join(DATA_DIR, 'dataAfterPre.csv'))

data=np.array(data)
y=data[:,0]
x_matrix=data[:,1:]

#--------------------------------------------split------------------------
x_train,x_valid,y_train,y_valid=train_test_split(x_matrix,y,test_size=0.1,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.1,random_state=42)

#--------------------------------------------pipeline------------------------
pipeline=Pipeline([
    ('scaler',RobustScaler()),
    ('poly',PolynomialFeatures(degree=3,include_bias=False)),
    
    ('model',Ridge(100))
])

pipeline.fit(x_train,y_train)

y_valid_predicted=pipeline.predict(x_valid)
y_test_predicted=pipeline.predict(x_test)
r2_valid=r2_score(y_valid,y_valid_predicted)
r2_test=r2_score(y_test,y_test_predicted)
r2_train=r2_score(y_train,pipeline.predict(x_train))

print(f'r2 train = {r2_train}\nr2 for validation data = {r2_valid}\nr2 for test data = {r2_test}')


