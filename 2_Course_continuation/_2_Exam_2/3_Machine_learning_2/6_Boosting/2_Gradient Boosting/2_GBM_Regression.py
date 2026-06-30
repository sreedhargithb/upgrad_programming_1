#!/usr/bin/env python
# coding: utf-8

# **STEP 1:   IMPORTING LIBRARIES**

# In[1]:


import numpy as np
import pandas as pd
import xgboost
import math
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from __future__ import division
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import explained_variance_score,mean_squared_error
from time import time
from sklearn.metrics import r2_score
import os
from sklearn.model_selection import train_test_split
#Machine Learning
from sklearn.ensemble import GradientBoostingRegressor


# In[2]:


data = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_2_Exam_2/3_Machine_learning_2/6_Boosting/2_Gradient%20Boosting/kc_house_data.csv')


# In[3]:


# Copying data to another dataframe df_train for our convinience so that original dataframe remain intact.
df_train=data.copy()
df_train.rename(columns ={'price': 'SalePrice'}, inplace =True)


# In[4]:


# Now lets see the first five rows of the data
data.head()


# **STEP 2:  DATA CLEANING AND PREPROCESSING**

# In[5]:


print(len(data))
# Check the number of features in the data set
print(len(data.columns))
# Check the data types of each column
print(data.dtypes)


# In[6]:


# Check any number of columns with NaN or missing values 
print(data.isnull().any().sum(), ' / ', len(data.columns))


# In[7]:


# Check any number of data points with NaN
print(data.isnull().any(axis=1).sum(), ' / ', len(data))


# **STEP 3 : FINDING CORRELATION**

# In[8]:


# As id and date columns are not important to predict price so we are discarding it for finding correlation
features = data.iloc[:,3:].columns.tolist()
target = data.iloc[:,2].name


# In[9]:


# Finding Correlation of price with other variables to see how many variables are strongly correlated with price
correlations = {}
for f in features:
    data_temp = data[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0]


# In[10]:


# Printing all the correlated features value with respect to price which is target variable
# Checking Corelation with price 
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]


# **STEP 4 : EDA or DATA VISUALIZATION **

# Let's explore the data

# In[11]:


var = 'sqft_living15'
data = pd.concat([data['price'], data[var]], axis=1)
data.plot.scatter(x=var, y='price', ylim=(3,9500000))


# In[12]:


var = 'bedrooms'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(14, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=4000000);


# In[13]:


var = 'bathrooms'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 20))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=3500000);


# In[14]:


var = 'sqft_living'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(3,8000000));


# In[15]:


var = 'floors'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 20))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=3500000);


# In[16]:


#Pairplots to visualize strong correlation
sns.set()
cols = ['SalePrice', 'sqft_living', 'grade', 'sqft_above', 'view', 'bathrooms','bedrooms','sqft_basement']
sns.pairplot(df_train[cols], height = 3.5)
plt.show();


# In[17]:


df_train.dtypes


# In[18]:


filtered_data = df_train[['sqft_living','grade', 'sqft_above', 'sqft_living15','bathrooms','view','sqft_basement','waterfront','yr_built','lat','bedrooms','long']]


# In[19]:


X = filtered_data.values
y = df_train.SalePrice.values


# In[20]:


filtered_data.dtypes


# In[21]:


filtered_data.dtypes


# **STEP 5 : SPLITTING DATA INTO TRAINING AND TESTING SET**

# In[22]:


X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2)


# **STEP 6: APPLYING MACHINE LEARNING MODEL**

# # GBM Regression
# 
# ##### The most important parameters -
# 
# 
# #####  1. loss - Loss function to be optimized. By default ‘ls’ refers to least squares regression
# 
# #####  2. learning_rate - Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators
# 
# ##### 3. n_estimators - No of estimators or Trees
# 
# #### 4. min_samples_leaf - The minimum number of samples required to be at a leaf node. A split point at any depth will only be considered if it leaves at least min_samples_leaf training samples in each of the left and right branches. This may have the effect of smoothing the model, especially in regression
# 
# 
# #### 5. min_samples_split - The minimum number of samples required to split an internal node
# 
# #### 6. subsample - The fraction of samples to be used for fitting the individual base learners

# In[23]:


gbm = GradientBoostingRegressor().fit(X_train, y_train)

pred = gbm.predict(X_test)

r2score = r2_score(pred,y_test)


# In[24]:


pred


# In[25]:


# Calculating R2 Score
r2score


# In[26]:


# Calculating Mean Sqaured Error & Root Mean Squared Error
mse = mean_squared_error(y_test, pred)

rmse = math.sqrt(mse)

print(rmse)


# In[27]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

