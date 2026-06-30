#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
marks = pd.read_csv('https://media-doselect.s3.amazonaws.com/generic/GZVBQG5pzJeNaL1ve50Rv9YNV/class-grades.csv')
print(marks.isnull().sum())


# In[2]:


import pandas as pd
marks = pd.read_csv('https://media-doselect.s3.amazonaws.com/generic/VodvGKvJAaeNrG7kvOQV38aog/class-grades.csv')

#Type your code here to remove rows with missing value equal to 5
marks = marks[marks.isnull().sum(axis=1)==5]
print(marks.isna().sum())


# In[3]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

