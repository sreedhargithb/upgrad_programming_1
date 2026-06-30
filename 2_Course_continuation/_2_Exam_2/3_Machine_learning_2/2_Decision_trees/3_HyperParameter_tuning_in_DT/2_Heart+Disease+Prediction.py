#!/usr/bin/env python
# coding: utf-8

# In[74]:


# Importing the required libraries
import pandas as pd, numpy as np
import matplotlib.pyplot as plt, seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[75]:


# Reading the csv file and putting it into 'df' object.
df = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_2_Exam_2/3_Machine_learning_2/2_Decision_trees/3_HyperParameter_tuning_in_DT/heart_v2.csv')


# In[76]:


df.columns


# In[77]:


df.head()


# In[78]:


# Putting feature variable to X
X = df.drop('heart disease',axis=1)

# Putting response variable to y
y = df['heart disease']


# In[ ]:





# In[79]:


from sklearn.model_selection import train_test_split


# In[80]:


X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
X_train.shape, X_test.shape


# Fitting the decision tree with default hyperparameters, apart from max_depth which is 3 so that we can plot and read the tree.

# In[81]:


from sklearn.tree import DecisionTreeClassifier


# In[82]:


dt = DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)


# In[83]:


get_ipython().system('pip install six')


# In[84]:


# Importing required packages for visualization
from IPython.display import Image  
from six import StringIO  
from sklearn.tree import export_graphviz
import pydotplus, graphviz


# In[85]:


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

# In[86]:


y_train_pred = dt.predict(X_train)
y_test_pred = dt.predict(X_test)


# In[87]:


from sklearn.metrics import confusion_matrix, accuracy_score


# In[88]:


print(accuracy_score(y_train, y_train_pred))
confusion_matrix(y_train, y_train_pred)


# In[89]:


print(accuracy_score(y_test, y_test_pred))
confusion_matrix(y_test, y_test_pred)


# Creating helper functions to evaluate model performance and help plot the decision tree

# In[90]:


def get_dt_graph(dt_classifier):
    dot_data = StringIO()
    export_graphviz(dt_classifier, out_file=dot_data, filled=True,rounded=True,
                    feature_names=X.columns, 
                    class_names=['Disease', "No Disease"])
    graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
    return graph


# In[91]:


def evaluate_model(dt_classifier):
    print("Train Accuracy :", accuracy_score(y_train, dt_classifier.predict(X_train)))
    print("Train Confusion Matrix:")
    print(confusion_matrix(y_train, dt_classifier.predict(X_train)))
    print("-"*50)
    print("Test Accuracy :", accuracy_score(y_test, dt_classifier.predict(X_test)))
    print("Test Confusion Matrix:")
    print(confusion_matrix(y_test, dt_classifier.predict(X_test)))


# ### Without setting any hyper-parameters

# In[92]:


dt_default = DecisionTreeClassifier(random_state=42)
dt_default.fit(X_train, y_train)


# In[93]:


gph = get_dt_graph(dt_default)
Image(gph.create_png())


# In[94]:


evaluate_model(dt_default)


# ### Controlling the depth of the tree

# In[95]:


get_ipython().run_line_magic('pinfo', 'DecisionTreeClassifier')


# In[96]:


dt_depth = DecisionTreeClassifier(max_depth=3)
dt_depth.fit(X_train, y_train)


# In[97]:


gph = get_dt_graph(dt_depth) 
Image(gph.create_png())


# In[98]:


evaluate_model(dt_depth)


# ### Specifying minimum samples before split

# In[99]:


dt_min_split = DecisionTreeClassifier(min_samples_split=20)
dt_min_split.fit(X_train, y_train)


# In[100]:


gph = get_dt_graph(dt_min_split) 
Image(gph.create_png())


# In[101]:


evaluate_model(dt_min_split)


# ### Specifying minimum samples in leaf node

# In[102]:


dt_min_leaf = DecisionTreeClassifier(min_samples_leaf=20, random_state=42)
dt_min_leaf.fit(X_train, y_train)


# In[103]:


gph = get_dt_graph(dt_min_leaf)
Image(gph.create_png())


# In[104]:


evaluate_model(dt_min_leaf)


# ### Using Entropy instead of Gini

# In[105]:


dt_min_leaf_entropy = DecisionTreeClassifier(min_samples_leaf=20, random_state=42, criterion="entropy")
dt_min_leaf_entropy.fit(X_train, y_train)


# In[106]:


gph = get_dt_graph(dt_min_leaf_entropy)
Image(gph.create_png())


# In[107]:


evaluate_model(dt_min_leaf_entropy)


# ### Hyper-parameter tuning

# In[108]:


dt = DecisionTreeClassifier(random_state=42)


# In[109]:


from sklearn.model_selection import GridSearchCV


# In[110]:


# Create the parameter grid based on the results of random search 
params = {
    'max_depth': [2, 3, 5, 10, 20],
    'min_samples_leaf': [5, 10, 20, 50, 100],
    'criterion': ["gini", "entropy"]
}


# In[111]:


# grid_search = GridSearchCV(estimator=dt, 
#                            param_grid=params, 
#                            cv=4, n_jobs=-1, verbose=1, scoring = "f1")


# In[112]:


# Instantiate the grid search model
grid_search = GridSearchCV(estimator=dt, 
                           param_grid=params, 
                           cv=4, n_jobs=-1, verbose=1, scoring = "accuracy")


# In[113]:


get_ipython().run_cell_magic('time', '', 'grid_search.fit(X_train, y_train)\n')


# In[115]:


score_df = pd.DataFrame(grid_search.cv_results_)
score_df.head()


# In[117]:


score_df.nlargest(5,"mean_test_score")


# In[119]:


grid_search.best_estimator_


# In[121]:


dt_best = grid_search.best_estimator_


# In[123]:


evaluate_model(dt_best)


# In[125]:


from sklearn.metrics import classification_report


# In[127]:


print(classification_report(y_test, dt_best.predict(X_test)))


# In[129]:


gph = get_dt_graph(dt_best)
Image(gph.create_png())


# In[131]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

