import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import datetime as dt
#------------------------------------------absolute path----------------


import os
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")
#------------------------------------------data Quality----------------
#1-completeness
#2-uniqueness
#3-timelinces
# 4-valid
# 5-consistency
#accuricy #

#encoding + corr + outliers + feature transformation
df=pd.read_csv(os.path.join(DATA_DIR, 'kc_house_data.csv'))

df.drop(['id','zipcode'],axis=1,inplace=True)

df['date']=df['date'].apply(lambda x:x.split('T')[0])
df['date']=pd.to_datetime(df['date'])
df['sale_year']=df['date'].dt.year
df['age']=df['sale_year']-df['yr_built']

# print(df.columns)
# sns.heatmap(df.corr(),annot=True)
# plt.show()

df.drop('sqft_above',axis=1,inplace=True) # hight corr with sqft_living leading to multicolinarity

df.drop(['sale_year', 'sqft_lot', 'sqft_lot15',  'condition','date'],axis=1,inplace=True) # low correlation with the price

df=df[~((df['bathrooms']==0)&(df['sqft_living']>1000))] #outliers
df=df[df['bedrooms']<=15] #outliers

df['sqft_living']=np.log1p(df['sqft_living']) #dealing with skewness 
df['basement_existant']=(df['sqft_basement']>0).astype(int) #label
df['sqft_living15']=np.log1p(df['sqft_living15']) #dealing with skewness 
df['price']=np.log1p(df['price'])  #dealing with skewness 
df['waterfront']=(df['waterfront']>0).astype(int) #label
df['yr_renovated']=(df['yr_renovated']>0).astype(int) # label



            



df.to_csv(os.path.join(DATA_DIR, 'dataAfterPre.csv'),index=False)