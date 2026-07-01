#!/usr/bin/env python
# coding: utf-8

# In[57]:


# Importing the required libraries
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[58]:


# Reading the csv file and putting it into 'df' object.
df = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_2_Exam_2/3_Machine_learning_2/2_Decision_trees/2_Algos_for_DT_construction/heart_v2.csv')


# In[59]:


print(df.columns)


# In[60]:


print(df.head())


# In[61]:


# Putting feature variable to X
X = df.drop('heart disease',axis=1)

# Putting response variable to y
y = df['heart disease']


# In[62]:


from sklearn.model_selection import train_test_split


# In[63]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape


# Fitting the decision tree with default hyperparameters, apart from max_depth which is 3 so that we can plot and read the tree.

# In[64]:


from sklearn.tree import DecisionTreeClassifier


# In[65]:


dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)


# In[66]:


get_ipython().system('pip install six pydotplus graphviz')
get_ipython().system('conda install -y graphviz')


# In[67]:


# Importing required packages for visualization
from IPython.display import Image
from six import StringIO
from sklearn.tree import export_graphviz
import pydotplus, graphviz


# In[68]:


# plotting tree with max_depth=3
dot_data = StringIO()

export_graphviz(dt, out_file=dot_data, filled=True, rounded=True,
                feature_names=X.columns,
                class_names=['No Disease', "Disease"])

graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
Image(graph.create_png())
#Image(graph.create_png(),width=800,height=900)
#graph.write_pdf("dt_heartdisease.pdf")


# #### Evaluating model performance

# In[69]:


y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)


# In[70]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[71]:


print(accuracy_score(y_train, y_train_pred))
confusion_matrix(y_train, y_train_pred)


# In[72]:


print(accuracy_score(y_test, y_test_pred))
confusion_matrix(y_test, y_test_pred)


# In[73]:


import datetime, pytz;
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

