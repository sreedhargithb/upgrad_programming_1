#!/usr/bin/env python
# coding: utf-8

# # K Nearest Neighbors

# In[1]:


import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.neighbors import KNeighborsClassifier
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


# # Finding the best k in KNN

# In[5]:


k_range = range(1, 11)
score1=[]
score2=[]
for k in k_range:
    knn=KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train,y_train)
    score1.append(knn.score(X_train,y_train))
    score2.append(knn.score(X_test,y_test))

get_ipython().run_line_magic('matplotlib', 'inline')
plt.plot(k_range,score1,label= 'Accuracy on training set')
plt.plot(k_range,score2,label= 'Accuracy on testing set')
plt.xlabel('Value of K in KNN')
plt.ylabel('Accuracy')
plt.legend()


# ### Here n in KNN is 7

# In[6]:


knn=KNeighborsClassifier(n_neighbors=7)
knn.fit(X_train,y_train)
print('Accuracy of KNN n=7 on the testing dataset is :{:.3f}'.format(knn.score(X_test,y_test)))


# In[7]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

