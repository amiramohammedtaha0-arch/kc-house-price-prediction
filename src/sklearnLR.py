import pandas as pd 
from modelImp import dataPrep
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#------------------------------------------absolute path----------------


import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

#splitting
#pipeline ==> polyfeatures --> scaling -->model
#-----------------------------------------data----------------------------
data=pd.read_csv(os.path.join(DATA_DIR, 'dataAfterPre.csv'))
y,x_matrix=dataPrep(data)
x_matrix=x_matrix[:,1:]


#--------------------------------------------split------------------------
x_train,x_valid,y_train,y_valid=train_test_split(x_matrix,y,test_size=0.1,random_state=42)
x_train,x_test,y_train,y_test=train_test_split(x_train,y_train,test_size=0.1,random_state=42)

#--------------------------------------------poly------------------------
poly=PolynomialFeatures(degree=3,include_bias=False)
x_train_poly=poly.fit_transform(x_train)
x_valid_poly=poly.transform(x_valid)
x_test_poly=poly.transform(x_test)

#--------------------------------------------poly------------------------
scaler=StandardScaler()
x_train_scaled=scaler.fit_transform(x_train_poly)
x_valid_scaled=scaler.transform(x_valid_poly)
x_test_scaled=scaler.transform(x_test_poly)

#--------------------------------------------model------------------------

model=LinearRegression()
model.fit(x_train_scaled,y_train)
y_valid_predicted=model.predict(x_valid_scaled)
y_test_predicted=model.predict(x_test_scaled)

r2_valid=r2_score(y_valid,y_valid_predicted)
r2_test=r2_score(y_test,y_test_predicted)


print(f'r2 for validation data = {r2_valid} and r2 for test data = {r2_test}')


