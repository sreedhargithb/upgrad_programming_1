#!/usr/bin/env python
# coding: utf-8

# 

# In[8]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

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


# In[9]:


get_ipython().system('pip install tensorflow')
import PIL #python imaging library enables python to perform different image operations
import tensorflow as tf #deep learning library enables to create neural networks
import matplotlib.pyplot as plt #for basic plotting


# In[10]:


import kagglehub

# Download latest version
path = kagglehub.dataset_download("paultimothymooney/chest-xray-pneumonia")

print("Path to dataset files:", path)
image = PIL.Image.open("/kaggle/input/chest-xray-pneumonia/chest_xray/train/PNEUMONIA/person1000_bacteria_2931.jpeg")


# In[11]:


print(image)


# In[12]:


image = PIL.Image.open("/kaggle/input/chest-xray-pneumonia/chest_xray/train/NORMAL/IM-0115-0001.jpeg")


# In[13]:


print(image)


# In[14]:


from tensorflow.keras.preprocessing.image import ImageDataGenerator #take whole datset as a input and create 3 types of other image generators - train, validate, test


# In[15]:


# to train the model
training_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/train/"
training_generator = ImageDataGenerator(rescale = 1/255) # rescale the images or divide each pixel to 255
data_train = training_generator.flow_from_directory(training_dir, target_size = (120, 120), batch_size = 8, class_mode = "binary") # load data from training directory to train the model


# In[16]:


# Define validation generator
valid_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/val/"
validation_generator = ImageDataGenerator(rescale = 1/255)
data_valid = validation_generator.flow_from_directory(valid_dir, target_size = (120, 120), batch_size = 8, class_mode = "binary") #load data from validation directory


# In[17]:


# Define the test generator
test_dir = "/kaggle/input/chest-xray-pneumonia/chest_xray/test/"
test_generator = ImageDataGenerator(rescale = 1/255)
data_test = test_generator.flow_from_directory(test_dir, target_size = (120, 120), batch_size = 8, class_mode = "binary")


# ## CNN
# 

# In[18]:


# build convolution neural network(CNN)
model = tf.keras.Sequential([tf.keras.layers.Conv2D(32, (3, 3), input_shape = (120, 120, 3), activation = "relu"),# #call sequential and give input of different types of layers
                            tf.keras.layers.MaxPooling2D(2, 2), # pooling layer define size of matrix obtained from convulation layer
                            tf.keras.layers.Conv2D(64, (3, 3), activation = "relu"), 
                            tf.keras.layers.MaxPooling2D(2, 2), 
                            tf.keras.layers.Conv2D(128, (3, 3), activation = "relu"), 
                            tf.keras.layers.MaxPooling2D(2, 2),
                            tf.keras.layers.Conv2D(256, (3, 3), activation = "relu"), 
                            tf.keras.layers.MaxPooling2D(2, 2),
                            tf.keras.layers.Conv2D(512, (3, 3), activation = "relu"),
                            tf.keras.layers.MaxPooling2D(2, 2),
                            tf.keras.layers.Flatten(),  # all the features should be converted to 1d array
                            tf.keras.layers.Dense(256, activation = "relu"),
                            tf.keras.layers.Dense(2, activation='softmax'),
                            tf.keras.layers.Dense(1, activation = "sigmoid")])  # Dense layer -> 1 what is extent that i/p image has lungs with pnuemonia and sigmoid for activation 
                                                    # define the no of filters the size of filters and shape of images


# In[19]:


model.summary()


# In[20]:


model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001), loss = "binary_crossentropy", metrics = ["acc"])


# 

# In[21]:


history = model.fit(data_train, epochs=2, validation_data=data_valid)


# In[22]:


model.evaluate(data_test)


# In[23]:


predictions = model.predict(data_test)


# In[24]:


x = next(data_test) 
for i in range(0, 1):
    image = x[i]
    for j in range(0, 8):
        plt.imshow(image[j])
        plt.show()
        print("The probablility of pnueonia is:", predictions[j])


# In[25]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:




