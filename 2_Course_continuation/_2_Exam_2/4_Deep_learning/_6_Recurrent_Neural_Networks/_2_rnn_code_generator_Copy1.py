#!/usr/bin/env python
# coding: utf-8

# # Text generation using RNN - Character Level (TO BE RUN IN GOOGLE COLAB)
# 
# To generate text using RNN, we need a to convert raw text to a supervised learning problem format.
# 
# Take, for example, the following corpus:
# 
# "Her brother shook his head incredulously"
# 
# First we need to divide the data into tabular format containing input (X) and output (y) sequences. In case of a character level model, the X and y will look like this:
# 
# |      X     |  Y  |
# |------------|-----|
# |    Her b   |  r  |
# |    er br   |  o  |
# |    r bro   |  t  |
# |     brot   |  h  |
# |    broth   |  e  |
# |    .....   |  .  |
# |    .....   |  .  |
# |    ulous   |  l  |
# |    lousl   |  y  |
# 
# Note that in the above problem, the sequence length of X is five characters and that of y is one character. Hence, this is a many-to-one architecture. We can, however, change the number of input characters to any number of characters depending on the type of problem.
# 
# A model is trained on such data. To generate text, we simply give the model any five characters using which it predicts the next character. Then it appends the predicted character to the input sequence (on the extreme right of the sequence) and discards the first character (character on extreme left of the sequence). Then it predicts again using the new sequence and the cycle continues until a fix number of iterations. An example is shown below:
# 
# Seed text: "incre"
# 
# |      X                                            |  Y                       |
# |---------------------------------------------------|--------------------------|
# |                        incre                      |    < predicted char 1 >  |
# |               ncre < predicted char 1 >              |    < predicted char 2 >  |
# |       cre< predicted char 1 > < predicted char 2 >   |    < predicted char 3 >  |
# |       re< predicted char 1 >< predicted char 2 > < predicted char 3 >   |    < predicted char 4 >  |
# |                      ...                          |            ...           |

# # Notebook Overview
# 1. Preprocess data
# 2. LSTM model
# 3. Generate code

# In[1]:


get_ipython().system('pip install gitpython')


# In[2]:


# import libraries
import warnings
warnings.filterwarnings("ignore")

import os
import re
import numpy as np
import random
import sys
import io
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Activation, LSTM
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import get_file


# # 1. Preprocess data

# We're going to build a C code generator by training an RNN on a huge corpus of C code (the linux kernel code). You can download the C code used as source text from the following link:
# https://github.com/torvalds/linux/tree/master/kernel
# 
# We have already downloaded the entire kernel folder and stored in a local directory

# ## Load C code

# In[3]:


import os
import tempfile
import git

# Define the repository and directory path
repo_url = "https://github.com/aqwertyuiop48/upgrad_programming"
# Use a portable path: Colab's /content when available, else system temp dir
if os.path.isdir("/content"):
    repo_path = "/content/upgrad_programming"
else:
    repo_path = os.path.join(tempfile.gettempdir(), "upgrad_programming")
subdir_path = "2_Course_continuation/_2_Exam_2/4_Deep_learning/_6_Recurrent_Neural_Networks"

# Clone the repository
if not os.path.exists(repo_path):
    git.Repo.clone_from(repo_url, repo_path)

# Change the working directory to the specific folder
os.chdir(os.path.join(repo_path, subdir_path))
print("Current working directory:", os.getcwd())


# In[4]:


# set path where C files reside

print("Current working directory:", os.getcwd())

path = r"linux_kernel"

os.chdir(path)

file_names = os.listdir()
print(file_names)


# In[5]:


# use regex to filter .c files
import re
c_names = ".*\.c$"

c_files = list()

for file in file_names:
    if re.match(c_names, file):
        c_files.append(file)

print(c_files)


# In[6]:


# load all c code in a list
full_code = list()
for file in c_files:
    code = open(file, "r", encoding='utf-8')
    full_code.append(code.read())
    code.close()


# In[7]:


# let's look at how a typical C code looks like
print(full_code[20])


# In[8]:


# merge different c codes into one big c code
text = "\n".join(full_code)
print("Total number of characters in entire code: {}".format(len(text)))


# In[9]:


# top_n: only consider first top_n characters and discard the rest for memory and computational efficiency
top_n = 400000
text = text[:top_n]


# ## Convert characters to integers

# In[10]:


# create character to index mapping
chars = sorted(list(set(text)))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))


# In[11]:


print("Vocabulary size: {}".format(len(chars)))


# ## Divide data in input (X) and output (y)

# ### Create sequences

# In[12]:


# define length for each sequence
MAX_SEQ_LENGTH = 50          # number of input characters (X) in each sequence
STEP           = 3           # increment between each sequence
VOCAB_SIZE     = len(chars)  # total number of unique characters in dataset

sentences  = []              # X
next_chars = []              # y

for i in range(0, len(text) - MAX_SEQ_LENGTH, STEP):
    sentences.append(text[i: i + MAX_SEQ_LENGTH])
    next_chars.append(text[i + MAX_SEQ_LENGTH])


# In[13]:


print('Number of training samples: {}'.format(len(sentences)))


# ## Create input and output using the created sequences
# 
# When you're not using the Embedding layer of the Keras as the very first layer, you need to convert your data in the following format:
# #### input shape should be of the form :  (#samples, #timesteps, #features)
# #### output shape should be of the form :  (#samples, #timesteps, #features)
# 
# ![Tensor shape](./jupyter resources/rnn_tensor.png)
# 
# #samples: the number of data points (or sequences)
# #timesteps: It's the length of the sequence of your data (the MAX_SEQ_LENGTH variable).
# #features: Number of features depends on the type of problem. In this problem, #features is the vocabulary size, that is, the dimensionality of the one-hot encoding matrix using which each character is being represented. If you're working with **images**, features size will be equal to: (height, width, channels), and the input shape will be (#training_samples, #timesteps, height, width, channels)

# In[14]:


# create X and y
X = np.zeros((len(sentences), MAX_SEQ_LENGTH, VOCAB_SIZE), dtype=bool)
y = np.zeros((len(sentences), VOCAB_SIZE), dtype=bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1


# In[15]:


print("Shape of X: {}".format(X.shape))
print("Shape of y: {}".format(y.shape))


# Here, X is reshaped to (#samples, #timesteps, #features). We have explicitly mentioned the third dimension (#features) because we won't use the Embedding() layer of Keras in this case since there are only 97 characters. Characters can be represented as one-hot encoded vector. There are no word embeddings for characters.

# # 2. LSTM

# In[16]:


from keras.optimizers import Adam

# define model architecture - using a two-layer LSTM with 128 LSTM cells in each layer
model = Sequential()
model.add(LSTM(128, input_shape=(MAX_SEQ_LENGTH, VOCAB_SIZE), return_sequences=True, dropout=0.5))
model.add(LSTM(128, dropout=0.5))
model.add(Dense(VOCAB_SIZE, activation = "softmax"))

optimizer = Adam(learning_rate=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics = ['acc'])


# In[17]:


# check model summary
model.summary()


# In[18]:


# fit model
model.fit(X, y, batch_size=128, epochs=20)


# # 3. Generate code

# Create a function that will make next character predictions based on temperature. If temperature is greater than 1, the generated characters will be more versatile and diverse. On the other hand, if temperature is less than one, the generated characters will be much more conservative.

# In[19]:


# define function to sample next word from a probability array based on temperature
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# In[20]:


np.random.multinomial(10, [0.05, 0.9, 0.05], size=2)


# In[21]:


# generate code

start_index = random.randint(0, len(text) - MAX_SEQ_LENGTH - 1) # pick random code to start text generation

for diversity in [0.5, 1.0, 1.5]:
        print('-'*50, 'diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + MAX_SEQ_LENGTH]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(1000):
            x_pred = np.zeros((1, MAX_SEQ_LENGTH, VOCAB_SIZE))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()


# In[22]:


# generate code

start_index = random.randint(0, len(text) - MAX_SEQ_LENGTH - 1) # pick random seed

for diversity in [0.5, 1.0, 1.5]:
        print('-'*50, 'diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + MAX_SEQ_LENGTH]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(1000):
            x_pred = np.zeros((1, MAX_SEQ_LENGTH, VOCAB_SIZE))
            for t, char in enumerate(sentence):
                x_pred[0, t, char_indices[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()


# In[23]:


import datetime, pytz;
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

