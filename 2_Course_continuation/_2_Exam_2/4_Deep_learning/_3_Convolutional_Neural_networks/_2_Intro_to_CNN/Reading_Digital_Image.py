#!/usr/bin/env python
# coding: utf-8

# Importing the libraries openCV and Matplotlib for reading and plotting images
# 

# <a href="/notebooks/1_Prep%20sessions/1_2_Python_basics.ipynb">To install Python packages</a>

# In[52]:


from keras.datasets import mnist
import numpy as np
import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Downloading the MNIST data

# In[53]:


(x_train, _), (x_test, _) = mnist.load_data()
print("The shape of x_train dataset is", x_train.shape)


# ## Reading greyscale image
# Loading first sample from MNIST dataset. Resizing the image to 18x18. 

# In[54]:


# selecting the first sample
x = x_train[1]
print("The dimension of x is 2D matrix as ", x.shape)
# Resizing the image
x = cv2.resize(x, (18,18))


# Plotting the image using Matplotlib

# In[55]:


plt.imshow(x, cmap='gray')


# You can see that height and width of the matrix is 18x18, same as height and width of above image. So, each pixel is represented by number. 

# In[56]:


print("The range of pixel varies between 0 to 255")
print("The pixel having black is more close to 0 and pixel which is white is more close to 255")
print(x)


# ## Reading colour image

# In[57]:


# Reading color image
get_ipython().system('wget -O cat.jpg https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_2_Exam_2/4_Deep_learning/_3_Convolutional_Neural_networks/_2_Intro_to_CNN/cat.jpg')
cat = cv2.imread('cat.jpg')
plt.imshow(cv2.cvtColor(cat, cv2.COLOR_BGR2RGB))


# In[58]:


print('The shape of image is ', cat.shape)


# ### Plotting the RGB channels of the image. 

# In[59]:


cat_r  = cv2.imread('cat.jpg')
cat_r[:,:,1:2] = 0
plt.imshow(cat_r)


# In[60]:


cat_g  = cv2.imread('cat.jpg')
cat_g[:,:,(0,2)] = 0
plt.imshow(cat_g)


# In[61]:


cat_b  = cv2.imread('cat.jpg')
cat_b[:,:,0:1] = 0
plt.imshow(cat_b)


# In[62]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

