#!/usr/bin/env python
# coding: utf-8

# In[30]:


get_ipython().system('pip install imblearn scikit-learn')


# In[31]:


import numpy as np 
import pandas as pd 


import plotly.offline as py
py.init_notebook_mode(connected=True)
import plotly.graph_objs as go
import plotly.tools as tls
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.model_selection import train_test_split


# Import and suppress warnings
import warnings
warnings.filterwarnings('ignore')


# # 1. Exploratory Data Analysis
# 
# Let us load in the dataset via the trusty Pandas package into a dataframe object which we call **attrition** and have a quick look at the first few rows

# In[32]:


attrition = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_2_Exam_2/3_Machine_learning_2/6_Boosting/2_Gradient%20Boosting/WA_Fn-UseC_-HR-Employee-Attrition.csv')
print(attrition.head())


# In[33]:


# Looking for NaN
print(attrition.isnull().any())


# In[34]:


# attrition.Age.fillna('')


# ### Correlation of Features
# 

# In[35]:


# attrition.corr()


# #  Feature Engineering & Categorical Encoding
# 
# Task of Feature engineering and numerically encoding the categorical values in our dataset.

# In[36]:


# attrition.shape


# In[37]:


print(attrition.dtypes)


# In[38]:


# Empty list to store columns with categorical data
categorical = []
for col, value in attrition.items():
    if value.dtype == 'object' or pd.api.types.is_string_dtype(value):
        categorical.append(col)

# Store the numerical columns in a list numerical
numerical = attrition.columns.difference(categorical)

print("Categorical columns:", categorical)
print("Numerical columns:", numerical)


# In[39]:


print(numerical)


# In[40]:


print(categorical)


# In[41]:


# Store the categorical data in a dataframe called attrition_cat
attrition_cat = attrition[categorical]
attrition_cat = attrition_cat.drop(['Attrition'], axis=1) # Dropping the target column


# In[42]:


print(attrition_cat)


# Applying the **get_dummies** method

# In[43]:


# How can you convert categorial or string or object data into Numerical Format ?

# Process of converting your cat data into numerical format - Encoding process 

# Encoding (15 More )

# Label Encoding 

# One Hot Encoding ( OHE)

# Cat_A 

# Male
#Female 
#Male
#Female
# Prefer_not_to_say
# Male 

# OHE 

           # Cat_A_Male    #Cat_A_Female   #Cat_A_Prefer_not_to_say
#1# Male      1             0                0 
#2#Female     0             1                0
#3#Male       1             0                0
#4#Female     0             1                0
#5# Prefer_not_to_say 0     0                1
#6# Male 



# Label Encoding 

# Cat_A 

# Male   2       
#Female 1
#Male 2
#Female 1
# Prefer_not_to_say 3
# Male 2

# Target Encoding 
# Mean Encoding 































# In[44]:


# Filter your object datatypes 

from sklearn import preprocessing

le = preprocessing.LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
le.transform(["tokyo", "tokyo", "paris","amsterdam"])

# list(le.classes_)


#0 ,1,2


# In[ ]:





# In[45]:


attrition_cat = pd.get_dummies(attrition_cat)
print(attrition_cat.head(3))


# In[46]:


# Store the numerical features to a dataframe attrition_num
attrition_num = attrition[numerical]


# let's concat numerical and caterogial dfs

# In[47]:


# Concat the two dataframes together columnwise
attrition_final = pd.concat([attrition_num, attrition_cat], axis=1)


# In[48]:


print(attrition_final.shape)


# **Target variable**
# 
# The target in this case is given by the column **Attrition** which contains categorical variables therefore requires numerical encoding. We numerically encode it by creating a dictionary with the mapping given as 1 : Yes and 0 : No

# In[49]:


# Define a dictionary for the target mapping
target_map = {'Yes':1, 'No':0}
# Use the pandas apply method to numerically encode our attrition target variable
target = attrition["Attrition"].apply(lambda x: target_map[x])
print(target.head(3))


# 
# **Splitting Data into Train and Test sets**
# 

# In[50]:


# Split data into train and test sets as well as for validation and testing
train, test, target_train, target_test = train_test_split(attrition_final, target, train_size= 0.75,random_state=0);


# #  Implementing Machine Learning Models
# 

# ## GBM Classifier
# 
# 

# ### 1.n_estimators - No of Trees in the Model
# 
# ### 2.max_features - The number of features to consider while searching for a best split.Thumb Rule to have Square root of no of Columns
# 
# ### 3.max_depth - Maximum Depth of Tree and can be used to control overfiting 
# 
# ### 4.min_samples_leaf - Minimum samples (or observations) required in a terminal node or leaf.In general we need to have lower values  for it for Imbalanced problems
# 
# ### 5.subsample- The fraction of samples to be used for fitting the individual base learners
# 
# ### 6.learning_rate - Learning rate shrinks the contribution of each tree by learning_rate. There is a trade-off between learning_rate and n_estimators

# In[51]:


gb = GradientBoostingClassifier(random_state=100) # default 
gb.get_params()


# In[52]:


# Fit the model to our train and target
gb.fit(train, target_train)
# Get our predictions
gb_predictions = gb.predict(test)


# In[53]:


gb_predictions_prob = gb.predict_proba(test)
print(gb_predictions_prob)


# In[54]:


# Gradient Boosting Parameters
# gb_params ={
#     'n_estimators': 500,   # no of Trees 
#     'learning_rate' : 0.2,
#     'max_depth': 11,
#     'min_samples_leaf': 2,
#     'subsample': 1,
#     'max_features' : 'sqrt',
#     'random_state' : 100,
#     'verbose': 0
# }

#gb = GradientBoostingClassifier(**gb_params) # After Doing HPT , we can pass the paramaters


# In[55]:


accuracy_score(target_test, gb_predictions)


# ### Feature Importance Gradient Boosting Model
# 

# In[56]:


print(gb.feature_importances_)


# In[57]:


# Scatter plot 
trace = go.Scatter(
    y = gb.feature_importances_,
    x = attrition_final.columns.values,
    mode='markers',
    marker=dict(
        sizemode = 'diameter',
        sizeref = 1.3,
        size = 12,
        color = gb.feature_importances_,
        colorscale='Portland',
        showscale=True
    ),
    text = attrition_final.columns.values
)
data = [trace]

layout= go.Layout(
    autosize= True,
    title= 'GBM Model Feature Importance',
    hovermode= 'closest',
     xaxis= dict(
         ticklen= 5,
         showgrid=False,
        zeroline=False,
        showline=False
     ),
    yaxis=dict(
        title= 'Feature Importance',
        showgrid=False,
        zeroline=False,
        ticklen= 5,
        gridwidth= 2
    ),
    showlegend= False
)
fig = go.Figure(data=data, layout=layout)
py.iplot(fig,filename='scatter')


# In[58]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

