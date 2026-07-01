#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ## Part 1: Import the Housing data and do feature transformations

# In[2]:


df= pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_2_Exam_2/4_Deep_learning/_1_2_Intro_to_neural_networks/_3_Neural_network_implementation_using_keras/house_price_full.csv')
print(df.head())


# In[3]:


X = df.copy()
# Remove target
Y = X.pop('price')

# perform a scaler transform of the input data
scaler = StandardScaler()
X = scaler.fit_transform(X)

# perform log transformation of target variable (For Sandeep: Is this needed?)
Y = np.log(Y)


# In[4]:


df_scaled = pd.DataFrame(X)
print(df_scaled)


# In[5]:


print(Y)


# ## Part 2: Create Model Using `keras`
# 
# ![](multiple_neurons.png)

# In[6]:


from tensorflow import keras


# In[7]:


model = keras.Sequential(
    [
        keras.layers.Dense(
            2, activation="sigmoid", input_shape=(X.shape[-1],)
        ),
        keras.layers.Dense(1, activation="linear")
    ]
)
model.summary()


# ```python
# def random_init_params():
#     w1 = tf.Variable(tf.random.uniform((2, 2)))
#     b1 = tf.Variable(tf.random.uniform((1, 2)))
#     w2 = tf.Variable(tf.random.uniform((2, 1)))
#     b2 = tf.Variable(tf.random.uniform((1, 1)))
#     return w1,b1,w2,b2
# 
# 
# def forward_prop(x, w1, b1, w2, b2):
#     z1 = tf.matmul(x,w1) + b1
#     h1 = tf.math.sigmoid(z1)
#     z2 = tf.matmul(h1,w2) + b2
#     h2 = z2
#     return h2
# ```

# In[8]:


model.compile(
    optimizer=keras.optimizers.SGD(), loss="mean_squared_error"
)


# ```python
# def train(x, y, w1, b1, w2, b2):
#     y_true = y
#     with tf.GradientTape() as g:
#         y_pred = forward_prop(x, w1, b1, w2, b2)
# 
#         # loss
#         loss = 0.5*(y_true - y_pred)** 2
#     
#     #Gradient calculation  
#     print("**************************************************")
#     print("GRADIENTS")
#     print("**************************************************")
#     gw1, gb1, gw2, gb2 = g.gradient(loss, [w1, b1, w2, b2])
#     print(" the gradient for 1st layer weights are:\n",gw1.numpy())
#     print("--------------------------------------------------")
#     print(" the gradient for 2nd layer weights are:\n",gw2.numpy())
#     print("--------------------------------------------------")
#     print(" the gradient for 1st layer bias are:\n",gb1.numpy())
#     print("--------------------------------------------------")
#     print(" the gradient for 2nd layer bias are:\n",gb2.numpy())
#     print("--------------------------------------------------")
# 
#     # Gradient descent:
#     lr=0.2
#     w1.assign_sub(lr*gw1)
#     b1.assign_sub(lr*gb1) 
#     w2.assign_sub(lr*gw2)
#     b2.assign_sub(lr*gb2)
#     print("**************************************************")
#     print("NEW UPDATES")
#     print("**************************************************")
#     print(" the updated 1st layer weights are:\n",w1.numpy())
#     print("--------------------------------------------------")
#     print(" the updated 2nd layer weights are:\n",w2.numpy())
#     print("--------------------------------------------------")
#     print(" the updated 1st layer bias are:\n",b1.numpy())
#     print("--------------------------------------------------")
#     print(" the updated 2nd layer bias are:\n",b2.numpy())
# 
# 
#     return w1, b1, w2, b2,loss
# 
# ```

# In[9]:


model.fit(X,Y.values,epochs=10,batch_size=32)


# In[10]:


print(model.predict(X)[:,0])


# In[11]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

