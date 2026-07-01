#!/usr/bin/env python
# coding: utf-8

# ## TO BE RUN 
# # **Loading libraries and data**

# In[6]:


import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns


from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score,confusion_matrix, classification_report
from sklearn.model_selection import GridSearchCV


# In[7]:


import kagglehub
# Download latest version
path = kagglehub.dataset_download("blastchar/telco-customer-churn")
print("Path to dataset files:", path)
df=pd.read_csv('/kaggle/input/telco-customer-churn/WA_Fn-UseC_-Telco-Customer-Churn.csv')
print(df.head())


# # **Understanding the data**

# In[8]:


print(df.shape)


# In[9]:


print(df.isnull().sum())


# In[10]:


print(df.info())


# # **Data Manipulation**

# In[11]:


df.drop('customerID',axis=1,inplace=True)


# In[12]:


print(df.head())


# In[13]:


df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')


# In[14]:


print(df['TotalCharges'].info())


# In[15]:


print(df['TotalCharges'].isnull().sum())


# In[16]:


df.dropna(axis=0,inplace=True)


# In[17]:


print(df.isnull().sum())


# In[18]:


for i in df.columns:
    print(df[i].value_counts())


# In[19]:


df['PaymentMethod']=df['PaymentMethod'].replace({'Bank transfer (automatic)':'Automatic','Credit card (automatic)':'Automatic'})
print(df['PaymentMethod'].value_counts())


# # **EDA**

# In[20]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='gender', hue='Churn', palette='viridis', edgecolor='black')

plt.title('gender Status vs Churn')
plt.xlabel('gender Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[21]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='SeniorCitizen', hue='Churn', palette='viridis', edgecolor='black')

plt.title('SeniorCitizen Status vs Churn')
plt.xlabel('SeniorCitizen Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[22]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Partner', hue='Churn', palette='viridis', edgecolor='black')

plt.title('Partner Status vs Churn')
plt.xlabel('Partner Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[23]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='Dependents', hue='Churn', palette='viridis', edgecolor='black')

plt.title('Dependents Status vs Churn')
plt.xlabel('Dependents Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[24]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='PhoneService', hue='Churn', palette='viridis', edgecolor='black')

plt.title('PhoneService Status vs Churn')
plt.xlabel('PhoneService Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[25]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='PaymentMethod', hue='Churn', palette='viridis', edgecolor='black')

plt.title('PaymentMethod Status vs Churn')
plt.xlabel('PaymentMethod Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[26]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='InternetService', hue='Churn', palette='viridis', edgecolor='black')

plt.title('InternetService Status vs Churn')
plt.xlabel('InternetService Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[27]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='MultipleLines', hue='Churn', palette='viridis', edgecolor='black')

plt.title('MultipleLines Status vs Churn')
plt.xlabel('MultipleLines Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[28]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='OnlineSecurity', hue='Churn', palette='viridis', edgecolor='black')

plt.title('OnlineSecurity Status vs Churn')
plt.xlabel('OnlineSecurity Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[29]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='OnlineBackup', hue='Churn', palette='viridis', edgecolor='black')

plt.title('OnlineBackup Status vs Churn')
plt.xlabel('OnlineBackup Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[30]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='TechSupport', hue='Churn', palette='viridis', edgecolor='black')

plt.title('TechSupport Status vs Churn')
plt.xlabel('TechSupport Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# In[31]:


plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='DeviceProtection', hue='Churn', palette='viridis', edgecolor='black')

plt.title('DeviceProtection Status vs Churn')
plt.xlabel('DeviceProtection Status')
plt.ylabel('Count')
plt.grid(True)
plt.show()


# # **Data Preprocessing**

# In[32]:


label_encoders = {}
for col in df.select_dtypes(include=['object']).columns:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  

print(df)


# In[33]:


print(df.info())


# In[34]:


corr_matrix = df.corr()

plt.figure(figsize=(20, 12))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5)

plt.title('Correlation Heatmap')
plt.show()


# In[35]:


x=df.drop(columns=['Churn','gender','MultipleLines','InternetService','StreamingTV','PaymentMethod'])
y=df['Churn']


# **Here I noticed an imbalance in the values in the Target column so I use SMOTE.**

# In[36]:


print(y.value_counts())


# In[37]:


smote = SMOTE(random_state=42,k_neighbors=15)
X_resampled, y_resampled = smote.fit_resample(x, y)

print(Counter(y_resampled))


# In[38]:


x_train, x_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.3, random_state=42, stratify=y_resampled)


# In[39]:


columns_to_scale = ['tenure','TotalCharges','MonthlyCharges']
scaler = StandardScaler()

x_train[columns_to_scale] = scaler.fit_transform(x_train[columns_to_scale])

x_test[columns_to_scale] = scaler.transform(x_test[columns_to_scale])


# In[40]:


print(y_test.shape)


# In[41]:


print(x_test.shape)


# In[42]:


print(x_train.shape)


# In[43]:


print(y_train.shape)


# # **Model**

# KNN

# In[44]:


model=KNeighborsClassifier(n_neighbors=17)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
accuracy = accuracy_score(y_test, y_pred)
print('Accuracy: ',accuracy)


# LogisticRegression

# In[45]:


log_reg = LogisticRegression()

log_reg.fit(x_train, y_train)
y_pred = log_reg.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)


# SVM

# In[46]:


svc_model = SVC(random_state = 1)
svc_model.fit(x_train,y_train)
predict_y = svc_model.predict(x_test)
accuracy_svc = svc_model.score(x_test,y_test)
print(f'SVM accuracy is :{accuracy_svc:.2f}')


# In[47]:


svm_model = SVC(kernel='linear')
svm_model.fit(x_train, y_train)

y_train_pred = svm_model.predict(x_train)
y_test_pred = svm_model.predict(x_test)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))
print("\nTraining Classification Report:\n", classification_report(y_train, y_train_pred))

print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))
print("\nTesting Classification Report:\n", classification_report(y_test, y_test_pred))


# RandomForestClassifier

# In[48]:


rf = RandomForestClassifier(n_estimators=250, max_depth=8)
rf.fit(x_train, y_train)

y_train_pred = rf.predict(x_train)
y_test_pred = rf.predict(x_test)

print("Training Accuracy:", accuracy_score(y_train, y_train_pred))

print("Testing Accuracy:", accuracy_score(y_test, y_test_pred))


# In[49]:


param_grid = {
    'n_estimators': [70, 100,120, 150,250],
    'max_depth': [5, 7,8, 9, 10],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)

best_rf = grid_search.best_estimator_
y_test_pred = best_rf.predict(x_test)

print("Optimized Testing Accuracy:", accuracy_score(y_test, y_test_pred))


# XGB

# In[50]:


from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score

xgb_model = XGBClassifier(n_estimators=70, max_depth=7, learning_rate=0.1, random_state=42)

xgb_model.fit(x_train, y_train)

y_train_pred = xgb_model.predict(x_train)
y_test_pred = xgb_model.predict(x_test)

train_acc = accuracy_score(y_train, y_train_pred)
test_acc = accuracy_score(y_test, y_test_pred)

print(f"Training Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy: {test_acc:.4f}")


# In[51]:


param_grid = {
    'n_estimators': [70, 200, 300],
    'max_depth': [4, 6, 7,8,9],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],

}

xgb_model = XGBClassifier(random_state=42)  

grid_search = GridSearchCV(xgb_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1, verbose=2)
grid_search.fit(x_train, y_train)

best_xgb = grid_search.best_estimator_
y_test_pred = best_xgb.predict(x_test)

# 
print("Optimized Testing Accuracy:", accuracy_score(y_test, y_test_pred))


# In[52]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

