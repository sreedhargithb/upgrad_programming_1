#!/usr/bin/env python
# coding: utf-8

# # Importing packages for XGBoost

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# # loading and seeing the description

# In[2]:


cancer=load_breast_cancer()
print(cancer.DESCR)


# In[3]:


print(cancer.feature_names)
print(cancer.target_names)


# # Splitting Datasets into Train & Test

# In[4]:


X_train,X_test,y_train,y_test =train_test_split(cancer.data,cancer.target, stratify=cancer.target, random_state=42)


# # Finding the best number of trees in XGBoost

# In[5]:


X_train,X_test,y_train,y_test =train_test_split(cancer.data,cancer.target, stratify=cancer.target, random_state=66)

tree_range = range(2, 30, 5)
score1=[]
score2=[]
for tree in tree_range:
    xgb=XGBClassifier(n_estimators=tree,eval_metric='mlogloss')
    xgb.fit(X_train,y_train)
    score1.append(xgb.score(X_train,y_train))
    score2.append(xgb.score(X_test,y_test))

get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(tree_range,score1,label= 'Accuracy on training set')
plt.plot(tree_range,score2,label= 'Accuracy on testing set')
plt.xlabel('Value of number of trees in XGboost')
plt.ylabel('Accuracy')
plt.legend()


# ### Here best number of trees in XGB is 25

# In[6]:


xgb=XGBClassifier(n_estimators=25,eval_metric='mlogloss')
xgb.fit(X_train,y_train)
print('Accuracy of XGB n_estimators=25 on the testing dataset is :{:.3f}'.format(xgb.score(X_test,y_test)))


# In[7]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

