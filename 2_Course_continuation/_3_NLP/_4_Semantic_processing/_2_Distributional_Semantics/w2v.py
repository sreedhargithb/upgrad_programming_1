#!/usr/bin/env python
# coding: utf-8

# # Word2Vec Tutorial with Gensim

# In[1]:


get_ipython().system('pip install gensim')
get_ipython().system('curl -L -o utils.py https://raw.githubusercontent.com/ContentUpgrad/semantic_processing/main/Distributional%20Semantics/codes/utils.py')


# In[2]:


# imports

import json
from collections import Counter
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
import utils

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


get_ipython().system('pip install kaggle')


# In[4]:


# Load and display data

get_ipython().system('wget https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_3_NLP/_4_Semantic_processing/_2_Distributional_Semantics/countries.json')

with open('countries.json', 'r') as fout:
    countries = json.load(fout)


# In[5]:


countries['India'][:20]


# In[6]:


print(' '.join(countries['India'])[:512] + ' ...')


# In[7]:


for i, (country, text) in enumerate(countries.items()):
    print(country)
    print(' '.join(text)[:512] + ' ...')
    print('-' * 100)
    if i >= 5:
        break


# ## Basic Word2Vec Usage

# In[8]:


# Create and train a simple model

model = Word2Vec(sentences=countries.values())


# In[9]:


# Check word similarities learnt by the model

model.wv.most_similar('India', topn=5)


# In[10]:


# Enable computation of loss

model = Word2Vec(
    sentences=countries.values(),
    compute_loss=True
)
model.get_latest_training_loss()


# ### Word2Vec options

# In[11]:


get_ipython().run_line_magic('pinfo', 'Word2Vec')


# ## Heuristics for Word2vec algorithms

# ### Determining size of the vocabulary

# In[12]:


# How many unique words in the vocabulary?

counter = Counter()
for words in countries.values():
    for word in words:
        counter.update([word])

print(len(counter))


# In[13]:


# Default vocabulary size of the original model

len(model.wv)


# In[14]:


# Retrain - increased vocabulary size, more epochs, larger word vectors

metric = utils.MetricCallback(every=1)
model = Word2Vec(
    sentences=countries.values(),
    vector_size=128,
    epochs=10,
    max_vocab_size=65536,
    compute_loss=True,
    callbacks=[metric]
)
plt.plot(metric.myloss)


# In[15]:


# Check similarities again

model.wv.most_similar('India')


# In[16]:


# Retrain - more epochs

metric = utils.MetricCallback(every=10)
model = Word2Vec(
    sentences=countries.values(),
    vector_size=128,
    epochs=100,
    max_vocab_size=65536,
    compute_loss=True,
    callbacks=[metric],
    min_alpha=0.001,
    workers=9
)
plt.plot(metric.myloss)


# In[17]:


model.wv.most_similar('India')


# In[18]:


# Examine the vector space

X = ['India', 'Pakistan', 'Bangladesh', 'France', 'England', 'Spain']
Y = ['Delhi', 'Islamabad', 'Dhaka', 'Paris', 'London', 'Madrid']
utils.plot_arrows(X, Y, model.wv)


# In[19]:


# Visualize vectors for all countries

utils.plot_vectors(countries, model)


# ## Word Analogies

# In[20]:


# India: Ganges -> Brazil: __ ?

model.wv.most_similar(positive=['Ganges', 'Brazil'], negative=['India'])


# In[21]:


# America: Washington -> France: __ ?

model.wv.most_similar(positive=['Washington', 'France'], negative=['America'])


# In[22]:


# India: Hindi -> Germany: __ ?

model.wv.most_similar(positive=['Hindi', 'Germany'], negative=['India'])


# In[23]:


# Save the model

model.save('wiki-countries.w2v')


# In[24]:


from gensim.models import KeyedVectors
model = KeyedVectors.load('wiki-countries.w2v')


# In[25]:


model


# In[26]:


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




