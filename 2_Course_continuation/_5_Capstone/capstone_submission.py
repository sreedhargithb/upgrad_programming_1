#!/usr/bin/env python
# coding: utf-8

# In[36]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

# import os
# for dirname, _, filenames in os.walk('/kaggle/input'):
#     for filename in filenames:
#         print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[37]:


import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[38]:


df=pd.read_csv('https://raw.githubusercontent.com/nsethi31/Kaggle-Data-Credit-Card-Fraud-Detection/refs/heads/master/creditcard.csv')
df.head(10)


# In[39]:


df.tail(10)


# In[40]:


df.shape


# In[41]:


df.info()


# In[42]:


df["Class"].value_counts()


# In[43]:


df = df.drop(['Time'],axis=1)
df.head()


# In[44]:


df.shape


# In[45]:


df.duplicated().any()


# In[46]:


df = df.drop_duplicates()
df.shape


# In[47]:


df["Class"].value_counts(normalize=True).plot(
    kind="bar", xlabel="Class", ylabel="Relative Frequency", title="Class Balance"
);


# In[48]:


df.hist(bins=30, figsize=(30, 30))


# In[49]:


df.describe()


# Scaling the dataset

# In[50]:


from sklearn.preprocessing import RobustScaler
new_df = df.copy()
new_df['Amount'] = RobustScaler().fit_transform(new_df['Amount'].to_numpy().reshape(-1, 1))
new_df['Amount'].hist();


# In[51]:


new_df['Amount'].describe()


# Copying the contents of the data into new_df

# In[52]:


new_df.head()


# Spliting the dataset into training and testing data

# In[53]:


X = new_df.drop('Class',axis=1)
y = new_df['Class']


# In[54]:


from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.20, random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


#  Building Models with the unbalanced dataset

# Logistic Regression

# In[55]:


from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)
logistic_model.score(X_train, y_train)


# In[56]:


# Predicting the result
y_pred = logistic_model.predict(X_test)


# In[57]:


from sklearn.metrics import accuracy_score,confusion_matrix,classification_report

ac = accuracy_score(y_test,y_pred)*100
cm = confusion_matrix(y_test,y_pred)
cr= classification_report(y_test,y_pred, target_names=['Not Fraud', 'Fraud'])
print("accuracy score:",ac)
print("confusion matrix:",cm)
print("classification report:",cr)


# Random Forest

# In[58]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=2, n_jobs=-1)
rf.fit(X_train, y_train)


# In[59]:


y_pred=rf.predict(X_test)


# In[60]:


ac = accuracy_score(y_test,y_pred)*100
cm = confusion_matrix(y_test,y_pred)
cr= classification_report(y_test,y_pred, target_names=['Not Fraud', 'Fraud'])
print("accuracy score:",ac)
print("confusion matrix:",cm)
print("classification report:",cr)


# Naive Bayes GaussianNB

# In[61]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

ac = accuracy_score(y_test,y_pred)*100
cm = confusion_matrix(y_test,y_pred)
cr= classification_report(y_test,y_pred, target_names=['Not Fraud', 'Fraud'])
print("accuracy score:",ac)
print("confusion matrix:",cm)
print("classification report:",cr)


# Decision Tree 

# In[62]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(random_state=42)
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

ac = accuracy_score(y_test,y_pred)*100
cm = confusion_matrix(y_test,y_pred)
cr= classification_report(y_test,y_pred, target_names=['Not Fraud', 'Fraud'])
print("accuracy score:",ac)
print("confusion matrix:",cm)
print("classification report:",cr)


# Balanced Dataset with OverSampling Technique

# In[63]:


X = new_df.drop('Class',axis=1)
y = new_df['Class']

print("X.shape: ", X.shape)
print("y.shape: ", y.shape)


# In[64]:


get_ipython().system('pip install imblearn')
from imblearn.over_sampling import SMOTE

X_res,y_res = SMOTE().fit_resample(X,y)
y_res.value_counts()


# Train Test Split on Balanced data

# In[65]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X_res,y_res,test_size=0.20,random_state=42)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)


# Logistic Regression on Balanced Data

# In[66]:


from sklearn.linear_model import LogisticRegression

logistic_model = LogisticRegression(max_iter=1000)
logistic_model.fit(X_train, y_train)

y_pred = logistic_model.predict(X_test)

ac = accuracy_score(y_test,y_pred)*100
cm = confusion_matrix(y_test,y_pred)
cr= classification_report(y_test,y_pred, target_names=['Not Fraud', 'Fraud'])
print("accuracy score:",ac)
print("confusion matrix:",cm)
print("classification report:",cr)


# Random Forest on Balanced Data

# In[67]:


from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(max_depth=2, n_jobs=-1)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

ac = accuracy_score(y_test,y_pred)*100
cm = confusion_matrix(y_test,y_pred)
cr= classification_report(y_test,y_pred, target_names=['Not Fraud', 'Fraud'])
print("accuracy score:",ac)
print("confusion matrix:",cm)
print("classification report:",cr)


# GaussianNB on Balanced data

# In[68]:


from sklearn.naive_bayes import GaussianNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)

y_pred = gnb.predict(X_test)

ac = accuracy_score(y_test,y_pred)*100
cm = confusion_matrix(y_test,y_pred)
cr= classification_report(y_test,y_pred, target_names=['Not Fraud', 'Fraud'])
print("accuracy score:",ac)
print("confusion matrix:",cm)
print("classification report:",cr)


# Decision Tree on Balanced data

# In[69]:


from sklearn.tree import DecisionTreeClassifier

dtc = DecisionTreeClassifier(max_depth=6, random_state=42)
dtc.fit(X_train, y_train)

y_pred = dtc.predict(X_test)

ac = accuracy_score(y_test,y_pred)*100
cm = confusion_matrix(y_test,y_pred)
cr= classification_report(y_test,y_pred, target_names=['Not Fraud', 'Fraud'])
print("accuracy score:",ac)
print("confusion matrix:",cm)
print("classification report:",cr)


# In[70]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

