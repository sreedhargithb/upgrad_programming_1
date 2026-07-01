#!/usr/bin/env python
# coding: utf-8

# # Flower Recognition CNN Keras (TO BE RUN IN GOOGLE COLAB)
# 

# 

# In[31]:


# Public no-auth mirror: TensorFlow's flower_photos (same 5 classes as the Kaggle
# alxmamaev/flowers-recognition dataset). Kaggle version needs auth (unavailable on CI).
import os as _os, urllib.request as _urlreq, tarfile as _tarfile
_FLOWERS_URL = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
_FLOWERS_TGZ = "/tmp/flower_photos.tgz"
_FLOWERS_BASE = "/tmp/flower_photos"
if not _os.path.exists(_FLOWERS_TGZ):
    print("Downloading flowers dataset (~218 MB)...")
    _urlreq.urlretrieve(_FLOWERS_URL, _FLOWERS_TGZ)
if not _os.path.isdir(_FLOWERS_BASE):
    with _tarfile.open(_FLOWERS_TGZ) as _t:
        _t.extractall("/tmp/")
print("Flowers dataset base:", _FLOWERS_BASE)


# In[32]:


print(_os.listdir(_FLOWERS_BASE))


# ## CONTENTS ::

# [ **1 ) Importing Various Modules**](#content1)

# [ **2 ) Preparing the Data**](#content2)

# [ **3 ) Modelling**](#content3)

# [ **4 ) Evaluating the Model Performance**](#content4)

# [ **5 ) Visualizing Predictons on the Validation Set**](#content5)

# 

# <a id="content1"></a>
# ## 1 ) Importing Various Modules.

# In[33]:


# Ignore  the warnings
import warnings
warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# data visualisation and manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import style
import seaborn as sns

#configure
# sets matplotlib to inline and displays graphs below the corressponding cell.
get_ipython().run_line_magic('matplotlib', 'inline')
style.use('fivethirtyeight')
sns.set(style='whitegrid',color_codes=True)

#model selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve,roc_auc_score
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

#preprocess.
from tensorflow.keras.preprocessing.image import ImageDataGenerator

#dl libraraies
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from keras.utils import to_categorical

# specifically for cnn
from keras.layers import Dropout, Flatten,Activation
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization

import tensorflow as tf
import random as rn

# specifically for manipulating zipped images and getting numpy arrays of pixel values of images.
import cv2
import numpy as np
from tqdm import tqdm
import os
from random import shuffle
from zipfile import ZipFile
from PIL import Image


# <a id="content2"></a>
# ## 2 ) Preparing the Data

# ## 2.1) Making the functions to get the training and validation set from the Images

# In[34]:


X=[]
Z=[]
IMG_SIZE=150
# TF's flower_photos uses plural folder names (roses/sunflowers/tulips) vs Kaggle's singular.
FLOWER_DAISY_DIR=_FLOWERS_BASE + '/daisy'
FLOWER_SUNFLOWER_DIR=_FLOWERS_BASE + '/sunflowers'
FLOWER_TULIP_DIR=_FLOWERS_BASE + '/tulips'
FLOWER_DANDI_DIR=_FLOWERS_BASE + '/dandelion'
FLOWER_ROSE_DIR=_FLOWERS_BASE + '/roses'


# In[35]:


def assign_label(img,flower_type):
    return flower_type


# In[36]:


def make_train_data(flower_type,DIR):
    for img in tqdm(os.listdir(DIR)):
        label=assign_label(img,flower_type)
        path = os.path.join(DIR,img)
        img = cv2.imread(path,cv2.IMREAD_COLOR)
        img = cv2.resize(img, (IMG_SIZE,IMG_SIZE))

        X.append(np.array(img))
        Z.append(str(label))




# In[37]:


make_train_data('Daisy',FLOWER_DAISY_DIR)
print(len(X))


# In[38]:


make_train_data('Sunflower',FLOWER_SUNFLOWER_DIR)
print(len(X))


# In[39]:


make_train_data('Tulip',FLOWER_TULIP_DIR)
print(len(X))


# In[40]:


make_train_data('Dandelion',FLOWER_DANDI_DIR)
print(len(X))


# In[41]:


make_train_data('Rose',FLOWER_ROSE_DIR)
print(len(X))


# ## 2.2 ) Visualizing some Random Images

# In[42]:


fig,ax=plt.subplots(5,2)
fig.set_size_inches(15,15)
for i in range(5):
    for j in range (2):
        l=rn.randint(0,len(Z))
        ax[i,j].imshow(X[l])
        ax[i,j].set_title('Flower: '+Z[l])

plt.tight_layout()


# ## 2.3 ) Label Encoding the Y array (i.e. Daisy->0, Rose->1 etc...) & then One Hot Encoding

# In[43]:


le=LabelEncoder()
Y=le.fit_transform(Z)
Y=to_categorical(Y,5)
X=np.array(X)
X=X/255


# ## 2.4 ) Splitting into Training and Validation Sets

# In[44]:


x_train,x_test,y_train,y_test=train_test_split(X,Y,test_size=0.25,random_state=42)


# ## 2.5 ) Setting the Random Seeds

# In[45]:


np.random.seed(42)
rn.seed(42)
tf.random.set_seed(42)


# In[45]:





# <a id="content3"></a>
# ## 3 ) Modelling

# ## 3.1 ) Building the ConvNet Model

# In[46]:


# # modelling starts using a CNN.

model = Sequential()
model.add(Conv2D(filters = 32, kernel_size = (5,5),padding = 'Same',activation ='relu', input_shape = (150,150,3)))
model.add(MaxPooling2D(pool_size=(2,2)))


model.add(Conv2D(filters = 64, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))


model.add(Conv2D(filters =96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Conv2D(filters = 96, kernel_size = (3,3),padding = 'Same',activation ='relu'))
model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(5, activation = "softmax"))


# ## 3.2 ) Using a LR Annealer

# In[47]:


batch_size=128
import os as _os_ci; epochs=int(_os_ci.environ.get('CI_EPOCHS', '50'))

from keras.callbacks import ReduceLROnPlateau
red_lr= ReduceLROnPlateau(monitor='val_acc',patience=3,verbose=1,factor=0.1)


# ## 3.3 ) Data Augmentation to prevent Overfitting

# In[48]:


datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=10,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.1, # Randomly zoom image
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(x_train)


# ## 3.4 ) Compiling the Keras Model & Summary

# In[49]:


model.compile(optimizer=Adam(learning_rate=0.001),loss='categorical_crossentropy',metrics=['accuracy'])


# In[50]:


model.summary()


# ## 3.5 ) Fitting on the Training set and making predcitons on the Validation set

# In[51]:


History = model.fit(datagen.flow(x_train,y_train, batch_size=batch_size),
                              epochs = epochs, validation_data = (x_test,y_test),
                              verbose = 1, steps_per_epoch=x_train.shape[0] // batch_size)
# model.fit(x_train,y_train,epochs=epochs,batch_size=batch_size,validation_data = (x_test,y_test))


# In[51]:





# <a id="content4"></a>
# ## 4 ) Evaluating the Model Performance

# In[52]:


plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# In[53]:


plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epochs')
plt.legend(['train', 'test'])
plt.show()


# <a id="content5"></a>
# ## 5 ) Visualizing Predictons on the Validation Set

# In[54]:


# getting predictions on val set.
pred=model.predict(x_test)
pred_digits=np.argmax(pred,axis=1)


# In[55]:


# now storing some properly as well as misclassified indexes'.
i=0
prop_class=[]
mis_class=[]

for i in range(len(y_test)):
    if(np.argmax(y_test[i])==pred_digits[i]):
        prop_class.append(i)
    if(len(prop_class)==8):
        break

i=0
for i in range(len(y_test)):
    if(not np.argmax(y_test[i])==pred_digits[i]):
        mis_class.append(i)
    if(len(mis_class)==8):
        break


# #### CORRECTLY CLASSIFIED FLOWER IMAGES

# In[56]:


warnings.filterwarnings('always')
warnings.filterwarnings('ignore')

# Assuming le is your LabelEncoder, prop_class is your list of indices, pred_digits is your list of predicted labels, and y_test is your test labels
count = 0
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

for i in range(2):
    for j in range(2):
        ax[i, j].imshow(x_test[prop_class[count]])
        predicted_flower = le.inverse_transform([pred_digits[prop_class[count]]])[0]  # Ensure this is a single value
        actual_flower = le.inverse_transform([np.argmax(y_test[prop_class[count]])])[0]  # Ensure this is a single value
        ax[i, j].set_title(f"Predicted Flower: {predicted_flower}\nActual Flower: {actual_flower}")
        plt.tight_layout()
        count += 1

plt.show()


# #### MISCLASSIFIED IMAGES OF FLOWERS

# In[57]:


# Assuming le is your LabelEncoder, mis_class is your list of misclassified indices, pred_digits is your list of predicted labels, and y_test is your test labels
count = 0
fig, ax = plt.subplots(2, 2, figsize=(10, 10))

for i in range(2):
    for j in range(2):
        ax[i, j].imshow(x_test[mis_class[count]])
        predicted_flower = le.inverse_transform([pred_digits[mis_class[count]]])[0]  # Ensure this is a single value
        actual_flower = le.inverse_transform([np.argmax(y_test[mis_class[count]])])[0]  # Ensure this is a single value
        ax[i, j].set_title(f"Predicted Flower: {predicted_flower}\nActual Flower: {actual_flower}")
        plt.tight_layout()
        count += 1

plt.show()


# In[58]:


import datetime, pytz;
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))


# #  THE END.

# 

# In[58]:




