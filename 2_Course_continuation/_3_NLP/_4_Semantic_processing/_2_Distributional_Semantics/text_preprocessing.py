#!/usr/bin/env python
# coding: utf-8

# # Text Preprocessing with Keras
# 
# ##### (TO BE DONE IN GOOGLE COLAB) https://colab.research.google.com/drive/1kwYhgcFZ0pyFY8EtAJJNzktiiUIMWMXK

# In[1]:


get_ipython().run_line_magic('pip', 'install tensor-sensor')


# In[2]:


# importing libraries

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D
import tsensor
import numpy as np


# ## Tokenization

# In[3]:


# Tokenising sentences
sentences = [
    'The quick brown fox jumps over the lazy dog.'
]
tokenizer = Tokenizer()
tokenizer.fit_on_texts(sentences)


# In[4]:


tokenizer.word_index


# In[5]:


train_sequence = tokenizer.texts_to_sequences(sentences)
train_sequence = np.array(train_sequence)
print(train_sequence)


# ## Creating Embedding Layer

# In[6]:


# Create a random embedding layer

embedding = Embedding(input_dim=len(train_sequence[0]), output_dim=128)


# In[7]:


# Get the embeddings of the train sample

train_sample = embedding(train_sequence)


# In[8]:


train_sequence.shape


# In[9]:


train_sample.shape


# In[10]:


with tsensor.explain(fontname='Hack', dimfontname='Hack'):
    train_sample = embedding(train_sequence)


# In[11]:


train_sample[0]


# ## Averaging across tokens

# In[12]:


GlobalAveragePooling1D()(train_sample)


# ![](images/Emb6.png)

# In[13]:


with tsensor.explain(fontname='Hack', dimfontname='Hack'):
    z = GlobalAveragePooling1D()(train_sample)


# ## Creating Word Embeddings for more than one sentence

# In[14]:


# More than one sentence

test_corpus = [
    'The quick brown fox jumps over the lazy dog.',
    'The quick brown fox.',
    'The lazy dog.',
    'The dog.',
    'Dog and the fox.',
    'Hello, world!'
]
encoded_sentences = tokenizer.texts_to_sequences(test_corpus)
for sentence, encoded_sentence in zip(test_corpus, encoded_sentences):
    print(sentence, encoded_sentence)


# ## Padding Sequences

# In[15]:


# Length of each sentence in the corpus

[len(sentence) for sentence in encoded_sentences]


# In[16]:


# Length of the longest sentence

max([len(sentence) for sentence in encoded_sentences])


# In[17]:


MAX_SEQUENCE_LENGTH = 9


# In[18]:


# Padding sequences that are shorter than the longest sequence

X = pad_sequences(encoded_sentences, maxlen=MAX_SEQUENCE_LENGTH)
X


# ## Embedding Layer

# In[19]:


# Training data with more than 1 sentences

X.shape


# In[20]:


# Embeddings of the larger corpus

X_embedded = embedding(X)


# In[21]:


X_embedded.shape


# In[22]:


X_embedded


# In[23]:


with tsensor.explain(fontname='Hack', dimfontname='Hack'):
    x_em = embedding(X)


# In[24]:


X.shape


# In[25]:


x_em.shape


# ## Averaging across tokens

# ![](images/Emb6.png)

# In[26]:


with tsensor.explain(fontname='Hack', dimfontname='Hack'):
    z = GlobalAveragePooling1D()(x_em)


# In[27]:


z.shape


# In[28]:


from datetime import datetime
import pytz

# Define the IST timezone
ist = pytz.timezone('Asia/Kolkata')

# Get the current time in UTC
utc_now = datetime.now(pytz.utc)

# Convert the current time to IST
ist_now = utc_now.astimezone(ist)

# Print the current time in IST
print("Current Time in IST:", ist_now.strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:




