#!/usr/bin/env python
# coding: utf-8
from __future__ import division

# **STEP 1:   IMPORTING LIBRARIES**

# In[33]:


import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
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
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, learning_curve, train_test_split


import xgboost as xgb


# In[34]:


get_ipython().system('pip install --upgrade --force-reinstall numpy scikit-learn xgboost joblib scipy')
get_ipython().system('pip install --upgrade imbalanced-learn shap hdbscan jax opencv-python')


# In[35]:


data = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_2_Exam_2/3_Machine_learning_2/6_Boosting/2_Gradient%20Boosting/kc_house_data.csv')


# In[36]:


# Copying data to another dataframe df_train for our convinience so that original dataframe remain intact.
df_train=data.copy()
df_train.rename(columns ={'price': 'SalePrice'}, inplace =True)


# In[37]:


# Now lets see the first five rows of the data
data.head()


# **STEP 2:  DATA CLEANING AND PREPROCESSING**

# In[38]:


print(len(data))
# Check the number of features in the data set
print(len(data.columns))
# Check the data types of each column
print(data.dtypes)


# In[39]:


# Check any number of columns with NaN or missing values 
print(data.isnull().any().sum(), ' / ', len(data.columns))


# In[40]:


# Check any number of data points with NaN
print(data.isnull().any(axis=1).sum(), ' / ', len(data))


# **STEP 3 : FINDING CORRELATION**

# In[41]:


# As id and date columns are not important to predict price so we are discarding it for finding correlation
features = data.iloc[:,3:].columns.tolist()
target = data.iloc[:,2].name


# In[42]:


# Finding Correlation of price with other variables to see how many variables are strongly correlated with price
correlations = {}
for f in features:
    data_temp = data[[f,target]]
    x1 = data_temp[f].values
    x2 = data_temp[target].values
    key = f + ' vs ' + target
    correlations[key] = pearsonr(x1,x2)[0]


# In[43]:


# Printing all the correlated features value with respect to price which is target variable
# Checking Corelation with price 
data_correlations = pd.DataFrame(correlations, index=['Value']).T
data_correlations.loc[data_correlations['Value'].abs().sort_values(ascending=False).index]


# **STEP 4 : EDA or DATA VISUALIZATION **

# Let's explore the data

# In[44]:


var = 'sqft_living15'
data = pd.concat([data['price'], data[var]], axis=1)
data.plot.scatter(x=var, y='price', ylim=(3,9500000))


# In[45]:


var = 'bedrooms'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(14, 6))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=4000000);


# In[46]:


var = 'bathrooms'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 20))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=3500000);


# In[47]:


var = 'sqft_living'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(3,8000000));


# In[48]:


var = 'floors'
data = pd.concat([df_train['SalePrice'], df_train[var]], axis=1)
f, ax = plt.subplots(figsize=(20, 20))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=3500000);


# In[49]:


#Pairplots to visualize strong correlation
sns.set()
cols = ['SalePrice', 'sqft_living', 'grade', 'sqft_above', 'view', 'bathrooms','bedrooms','sqft_basement']
sns.pairplot(df_train[cols], height = 3.5)
plt.show();


# In[50]:


df_train.dtypes


# In[51]:


filtered_data = df_train[['sqft_living','grade', 'sqft_above', 'sqft_living15','bathrooms','view','sqft_basement','waterfront','yr_built','lat','bedrooms','long']]


# In[52]:


X = filtered_data.values
y = df_train.SalePrice.values


# In[53]:


filtered_data.dtypes


# In[54]:


filtered_data.dtypes


# **STEP 5 : SPLITTING DATA INTO TRAINING AND TESTING SET**

# In[55]:


X_train, X_test, y_train, y_test = train_test_split(X, y ,test_size=0.2)


# **STEP 6: APPLYING MACHINE LEARNING MODEL**

# # Xgboost Regression
# 
# ##### The most important parameters -
# 
# 
# ### 1.n_estimators - No of Trees in the Model
# 
# ### 2.max_leaf_nodes = The maximum number of terminal nodes or leaves in a tree. If this is defined, max_depth will be ignored
# 
# ### 3.min_child_weight - Defines the minimum sum of weights of all observations required in a child.
# 
# ### 4.max_depth - Maximum Depth of Tree and can be used to control overfiting 
# 
# ### 5.subsample- The fraction of samples to be used for fitting the individual base learners
# 
# ### 6.learning_rate - Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators

# In[56]:


xgb_reg = xgb.XGBRegressor(n_jobs = -1) # default paramters
xgb_reg.get_params()


# In[57]:


xgb_reg.fit(X_train, y_train)


# In[58]:


pred = xgb_reg.predict(X_test)

r2score = r2_score(pred,y_test)


# In[59]:


pred


# In[60]:


# Calculating R2 Score
r2score


# In[61]:


# Calculating Mean Sqaured Error & Root Mean Squared Error
mse = mean_squared_error(y_test, pred)

rmse = math.sqrt(mse)

print(rmse)


# # HPT  - Random Search for Xgboost Regression

# In[62]:


# A parameter grid for XGBoost
params = {
        'n_estimators' : [100, 200, 500, 750], # no of trees 
        'learning_rate' : [0.01, 0.02, 0.05, 0.1, 0.25],  # eta
        'min_child_weight': [1, 5, 7, 10],
        'gamma': [0.1, 0.5, 1, 1.5, 5],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'max_depth': [3, 4, 5, 10, 12]
        }

folds = 3

param_comb = 100

random_search = RandomizedSearchCV(xgb_reg, param_distributions=params, n_iter=param_comb, n_jobs=-1, cv=3, verbose=3, random_state=42)


# In[63]:


random_search.fit(X_train, y_train)


# In[64]:


pred_hpt = random_search.predict(X_test)

r2score = r2_score(pred_hpt,y_test)


# In[65]:


pred_hpt


# In[66]:


r2score


# In[67]:


# Calculating Mean Sqaured Error & Root Mean Squared Error
mse = mean_squared_error(y_test, pred_hpt)

rmse = math.sqrt(mse)

print(rmse)


# In[68]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

