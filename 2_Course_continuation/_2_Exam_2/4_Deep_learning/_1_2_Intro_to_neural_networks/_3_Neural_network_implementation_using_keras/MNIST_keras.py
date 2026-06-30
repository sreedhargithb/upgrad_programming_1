#!/usr/bin/env python
# coding: utf-8

# ### Importing libraries

# In[15]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Loading data

# In[ ]:





# In[16]:


data = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_2_Exam_2/4_Deep_learning/_1_2_Intro_to_neural_networks/_3_Neural_network_implementation_using_keras/train.csv')
X_test = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_2_Exam_2/4_Deep_learning/_1_2_Intro_to_neural_networks/_3_Neural_network_implementation_using_keras/test.csv")


# In[17]:


data.head(5)


# ### Extracting feature variables and target variable

# In[18]:


data_y = data['label'].values
data_x = data.drop('label',axis=1)


# In[19]:


data_x.head(2)


# ### Data visualization

# In[20]:


img = data_x.iloc[2].values.reshape((28,28))
plt.imshow(img,cmap='gray')


# In[21]:


set(data_y)


# In[22]:


X_train, X_val, y_train, y_val = train_test_split(data_x, data_y, test_size = 0.1, random_state=42)


# ### Model Building

# In[23]:


model = keras.Sequential([
    keras.layers.Dense(128, activation=tf.nn.relu, input_shape=(784,)),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(128, activation=tf.nn.relu),
    keras.layers.Dense(10, activation=tf.nn.softmax)])


# In[24]:


model.compile(optimizer='adam',
             loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


# ### Model training

# In[25]:


model.fit(X_train, y_train, batch_size=64, epochs=5, validation_data=(X_val, y_val))


# ### Model evaluation

# In[26]:


model.predict(X_test).argmax(axis=1)


# In[27]:


img = X_test.iloc[1].values.reshape((28,28))
plt.imshow(img,cmap='gray')
print("The predicted label is:" ,model.predict(X_test).argmax(axis=1)[1])


# Further ahead: You can try increasing the model layers and play with the hyperparameters to increase model's performance

# In[28]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

