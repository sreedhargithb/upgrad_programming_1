#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# ### Part 1: Import the Housing data and do feature transformations

# In[2]:


df= pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_2_Exam_2/4_Deep_learning/_1_2_Intro_to_neural_networks/_1_Feedforward_neural_networks/house_price_full.csv')
df.head()


# In[ ]:


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
df_scaled


# In[ ]:


Y


# In[6]:


#Taking 1 sample: x0,x1
x1, x2 = df_scaled.iloc[0]


# In[7]:


x = tf.constant([[x1,x2]], dtype=tf.float32)
x


# ### Part2: Forward Propagation with a single Neuron
# 
# The simplest way to describe a neural network is that we have some inputs , which get combined into an auxilliary variable . The auxilliary variable is passed through the activation function  and the result is the output.
# 
# Here is another image showing each step.
# ![](neuron.png)
# Notice that the inputs are linearly combined according to some weights  and a bias . This transformation is also sometimes called an affine transformation. The perceptron transforms the weighted inputs according to the rule of the activation function. For a single perceptron, the output  is just the output from the perceptron. The linear transformation and activation of the neuron occurs within a single layer of the network (shown in the dotted box).
# 
# Let's see what the single-layer, single neuron network give us. We have a couple of choices to make:
# 
# We must choose some weights and some biases
# We must choose an activation function
# For now, we will manually specify the weights and biases.
# 
# We choose a sigmoid activation function

# In[8]:


#weights
w1 = tf.Variable([0.2], dtype=tf.float32)
w2 = tf.Variable([0.15], dtype=tf.float32)
#bias
b = tf.Variable([0.1], dtype=tf.float32)


# In[9]:


#Cumulative input
z = b + w1*x1 +w2*x2
h = tf.math.sigmoid(z)
print("The output from the first neuron is",h)


# ### Part3: Forward Propagation with multiple neurons
# 
# ![](multiple_neurons.png)

# In[ ]:


## layer1 weights
# neuron1
b1 = tf.Variable([0.1])
w11 = tf.Variable([0.2])
w12 = tf.Variable([0.15])
#neuron2
b2 = tf.Variable([0.25])
w21 = tf.Variable([0.5])
w22 = tf.Variable([0.6])


# In[11]:


## forward pass
# neuron 1
z1 = b1+w11*x1+w12*x2
h1 = tf.math.sigmoid(z1)
print("The output from the first neuron is",h1)


# In[12]:


## forward pass
# neuron 2
z2 = b2+w21*x1+w22*x2
h2 = tf.math.sigmoid(z2)
print("The output from the second neuron is",h2)


# In[13]:


## layer2 weights
b1 = tf.Variable([0.4])
w11 = tf.Variable([0.3])
w12 = tf.Variable([0.2])


# In[ ]:


## forward pass
# second layer
z1 = b1+w11*h1+w12*h2
h1 = z1
print("The output from the first neuron is",h1)


# In[15]:


y_true = Y[0]
y_pred = h1.numpy()


# In[ ]:


#loss
L = 0.5*(y_true - y_pred)**2
print("The MSE error is",L)


# ## Part 4: Forward pass matrix multiplication
# ![](multiple_neurons.png)
# 
# ![](Matrix.gif)
# 
# This network can be described as follows:
# 
# - Input vector = $X = (x1,x2)$
# - Weight Matrix (hidden layer) = $$W^1 = \begin{bmatrix}
# w^1_{11}&&w^1_{12}\\
# w^1_{21}&&w^1_{22}\\
# \end{bmatrix}
# $$
# *note the subscripts are being mapped to weights in the figure
# 
# - Bias/offset Matrix (hidden layer) = $$
# B^1_0 = \begin{bmatrix}
# b^1_{1}\\
# b^1_{2}\\
# \end{bmatrix}
# $$
# 
# Now the forward pass for the hidden layer can be described as 
# 
# $$W^1 \times X^T + B^1_0= Z^1 = \begin{bmatrix}
# z^1_{1}\\
# z^1_{2}\\
# \end{bmatrix}
# $$
# 
# Applying the activation function $f$ over the matrix $Z$ will complete the forward pass.
# 
# $$f(W^1 \times X^T + B^1_0)= f(Z^1) = f(\begin{bmatrix}
# z^1_{1}\\
# z^1_{2}\\
# \end{bmatrix}) = 
# \begin{bmatrix}
# f(z^1_{1})\\
# f(z^1_{2})\\
# \end{bmatrix}
# =
# \begin{bmatrix}
# h^1_1\\
# h^1_2\\
# \end{bmatrix}
# = H^1
# $$
# 
# For the output layer:
# 
# - The weight matrix is $$W^2 = \begin{bmatrix}
# w^2_{11}&&w^2_{12}\\
# \end{bmatrix}
# $$
# 
# - The bias/offset matrix is $$B^2_0 = \begin{bmatrix}
# b^2_{1}\\
# \end{bmatrix}
# $$
# 
# Now the forward pass can be written as:
# 
# 
# $$ B_0^2+W^2 \times H^1$$
# 

# In[17]:


## layer 1 weights
W1 = tf.Variable([[0.2, 0.15],
                     [0.5, 0.6]], dtype=tf.float32)
## layer 1 bias
B1 = tf.Variable([[0.1],
                [0.25]], dtype=tf.float32)


# In[18]:


## layer 2 weights
W2 = tf.Variable([[0.3, 0.2]], dtype=tf.float32)
#bias
B2 = tf.Variable([0.4], dtype=tf.float32)


# In[19]:


## data
X = tf.constant([[x1,x2]], dtype=tf.float32)


# In[20]:


## forward pass layer 1
Z1 = tf.matmul(W1, tf.transpose(X)) + B1
H1 = tf.math.sigmoid(Z1)
print(H1)


# In[21]:


## forward pass layer 2
Z2 = tf.matmul(W2,H1)+B2


# In[22]:


Z2


# In[23]:


y_pred = Z2.numpy()
loss = 0.5*(y_true-y_pred)**2
print(loss)


# ## Part5: Random Weight Initialization
# 
# ![](multiple_neurons.png)

# In[24]:


def random_init_params():
    w1 = tf.Variable(tf.random.uniform((2, 2)))
    b1 = tf.Variable(tf.random.uniform((1, 2)))
    w2 = tf.Variable(tf.random.uniform((2, 1)))
    b2 = tf.Variable(tf.random.uniform((1, 1)))
    return w1,b1,w2,b2


# In[25]:


x = tf.constant([[x1,x2]], dtype=tf.float32)
y = Y[0]
w1,b1,w2,b2 = random_init_params()


# In[26]:


print(" the initial 1st layer weights are:\n",w1.numpy())
print("--------------------------------------------------")
print(" the initial 2nd layer weights are:\n",w2.numpy())
print("--------------------------------------------------")
print(" the initial 1st layer bias are:\n",b1.numpy())
print("--------------------------------------------------")
print(" the initial 2nd layer bias are:\n",b2.numpy())


# In[ ]:


def forward_prop(x, w1, b1, w2, b2):
    z1 = tf.matmul(x,w1) + b1
    h1 = tf.math.sigmoid(z1)
    z2 = tf.matmul(h1,w2) + b2
    h2 = z2
    return h2


# In[28]:


y_pred = forward_prop(x, w1, b1, w2, b2)
#loss
L = 0.5*(y - y_pred)**2
print("The MSE error is",L)


# ## Part6: Backpropagation
# 
# Find the value of x that minimises $y = x^2+4x$
# 
# Gradient descent update equation
# 
# $x_{new} := x_{old}-\eta\frac{dy}{dx}$

# In[29]:


x = tf.Variable(0.0) ## add gradient tape
lr = eta = 0.1


# In[30]:


with tf.GradientTape() as tape:
    y = x**2+4*x
grad = tape.gradient(y,x) ## dy/dx


# In[31]:


grad.numpy() #dy/dx = 2x+4, x=0 => dy/dx = 4


# In[32]:


x.assign_sub(lr*grad) ## x_new = x_old -lr*dy/dx


# In[33]:


x.numpy()


# In[34]:


## full loop
x = tf.Variable(0.0) ## add gradient tape
lr = eta = 0.1
for i in range(10):
    with tf.GradientTape() as tape:
        y = x**2+4*x
    grad = tape.gradient(y,x)
    x.assign_sub(lr*grad)
    print(x.numpy())


# 
# ![](gradients.png)

# In[35]:


x = tf.constant([[x1,x2]], dtype=tf.float32)
y = Y[0]

def random_init_params():
    w1 = tf.Variable(tf.random.uniform((2, 2)))
    b1 = tf.Variable(tf.random.uniform((1, 2)))
    w2 = tf.Variable(tf.random.uniform((2, 1)))
    b2 = tf.Variable(tf.random.uniform((1, 1)))
    return w1,b1,w2,b2

def forward_prop(x, w1, b1, w2, b2):
    z1 = tf.matmul(x,w1) + b1
    h1 = tf.math.sigmoid(z1)
    z2 = tf.matmul(h1,w2) + b2
    h2 = z2
    return h2


# In[36]:


w1,b1,w2,b2 = random_init_params()


# In[ ]:


with tf.GradientTape() as tape:
    y_pred = forward_prop(x,w1,b1,w2,b2)
    loss = 0.5*(y-y_pred)**2


# In[38]:


gw1, gb1, gw2, gb2 = tape.gradient(loss, [w1, b1, w2, b2])


# In[39]:


gw1


# In[ ]:


gb1


# In[ ]:


gw2


# In[ ]:


gb2


# In[ ]:


lr=0.01
print(f"Value of w1 before gradient update is {w1}")
w1.assign_sub(lr*gw1)
print(f"Value of w1 after gradient update is {w1}")


# In[44]:


lr=0.01
print(f"Value of b1 before gradient update is {b1}")
b1.assign_sub(lr*gb1)
print(f"Value of w1 after gradient update is {b1}")


# In[ ]:


def train(x, y, w1, b1, w2, b2):
    y_true = y
    with tf.GradientTape() as g:
        y_pred = forward_prop(x, w1, b1, w2, b2)

        # loss
        loss = 0.5*(y_true - y_pred)** 2

    #Gradient calculation  
    print("**************************************************")
    print("GRADIENTS")
    print("**************************************************")
    gw1, gb1, gw2, gb2 = g.gradient(loss, [w1, b1, w2, b2])
    print(" the gradient for 1st layer weights are:\n",gw1.numpy())
    print("--------------------------------------------------")
    print(" the gradient for 2nd layer weights are:\n",gw2.numpy())
    print("--------------------------------------------------")
    print(" the gradient for 1st layer bias are:\n",gb1.numpy())
    print("--------------------------------------------------")
    print(" the gradient for 2nd layer bias are:\n",gb2.numpy())
    print("--------------------------------------------------")

    # Gradient descent:
    lr=0.2
    w1.assign_sub(lr*gw1)
    b1.assign_sub(lr*gb1) 
    w2.assign_sub(lr*gw2)
    b2.assign_sub(lr*gb2)
    print("**************************************************")
    print("NEW UPDATES")
    print("**************************************************")
    print(" the updated 1st layer weights are:\n",w1.numpy())
    print("--------------------------------------------------")
    print(" the updated 2nd layer weights are:\n",w2.numpy())
    print("--------------------------------------------------")
    print(" the updated 1st layer bias are:\n",b1.numpy())
    print("--------------------------------------------------")
    print(" the updated 2nd layer bias are:\n",b2.numpy())


    return w1, b1, w2, b2,loss


# In[46]:


w1,b1,w2,b2 = random_init_params()
w1, b1, w2, b2,loss = train(x, y, w1, b1, w2, b2)


# In[47]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

