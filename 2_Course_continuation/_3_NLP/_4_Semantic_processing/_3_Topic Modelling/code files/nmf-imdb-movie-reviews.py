#!/usr/bin/env python
# coding: utf-8

# # Inferring Topics from IMDB Reviews

# In[9]:


import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
import pandas as pd
import matplotlib.pyplot as plt


# ## Exploring the Dataset: [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz)

# In[10]:


# Download the archive (approx 80MB)
get_ipython().system('wget https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz')
get_ipython().system('tar -xzf aclImdb_v1.tar.gz')
ROOT = './aclImdb/train/pos/'


# In[11]:


reviews = []
for file in os.listdir(ROOT):
    path = os.path.join(ROOT, file)
    if os.path.isfile(path):
        with open(path, 'r') as fin:
            reviews.append(fin.read())


# In[12]:


len(reviews)


# In[13]:


for i in range(3):
    print(reviews[i])
    print('=' * 150)


# ## Feature Extraction

# In[14]:


vect = TfidfVectorizer(stop_words='english')
X = vect.fit_transform(reviews)

pd.DataFrame(X.toarray(), columns=vect.get_feature_names_out())


# ## NMF Decomposition

# In[15]:


N_TOPICS = 15
nmf = NMF(n_components=N_TOPICS)
W = nmf.fit_transform(X)  # Document-topic matrix
H = nmf.components_       # Topic-term matrix


# In[16]:


# Top 10 words per topic

words = np.array(vect.get_feature_names_out())
topic_words = pd.DataFrame(np.zeros((N_TOPICS, 10)), index=[f'Topic {i + 1}' for i in range(N_TOPICS)],
                           columns=[f'Word {i + 1}' for i in range(10)]).astype(str)
for i in range(N_TOPICS):
    ix = H[i].argsort()[::-1][:10]
    topic_words.iloc[i] = words[ix]

topic_words


# In[17]:


# Create a topic mapping

topic_mapping = {
    'Topic 4': 'TV',
    'Topic 7': 'War',
    'Topic 8': 'Comedy',
    'Topic 12': 'Book Adaptation',
    'Topic 13': 'Horror',
    'Topic 15': 'Martial Arts / Action'
}


# In[18]:


# Recall the document-topic matrix, W

W = pd.DataFrame(W, columns=[f'Topic {i + 1}' for i in range(N_TOPICS)])
W['max_topic'] = W.apply(lambda x: topic_mapping.get(x.idxmax()), axis=1)
W[pd.notnull(W['max_topic'])].head(10)


# In[19]:


reviews[58]


# In[20]:


# Frobenius norm

import numpy as np

print("Frobenius norm and the condition number:")
print(np.linalg.norm([[1,1,1],[3,4,1],[4,1,2]], 'fro'))
print(np.linalg.cond([[1,1,1],[3,4,1],[4,1,2]], 'fro'))


# In[21]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

