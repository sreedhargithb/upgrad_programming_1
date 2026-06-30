#!/usr/bin/env python
# coding: utf-8

# ## Lemmatization

# In[1]:


### import necessary libraries
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# In[2]:


text = "Very orderly and methodical he looked, with a hand on each knee, and a loud watch ticking a sonorous sermon under his flapped newly bought waist-coat, as though it pitted its gravity and longevity against the levity and evanescence of the brisk fire."
print(text)


# In[3]:


# tokenise text
tokens = word_tokenize(text)


# In[4]:


import nltk
nltk.download('wordnet')

wordnet_lemmatizer = WordNetLemmatizer()
lemmatized = [wordnet_lemmatizer.lemmatize(token) for token in tokens]
print(lemmatized)


# ### Let's compare stemming and lemmatization

# In[5]:


from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()
stemmed = [stemmer.stem(token) for token in tokens]
print(stemmed)


# In[6]:


import pandas as pd
df = pd.DataFrame(data={'token': tokens, 'stemmed': stemmed, 'lemmatized': lemmatized})
df = df[['token', 'stemmed', 'lemmatized']]
df[(df.token != df.stemmed) | (df.token != df.lemmatized)]


# Let's compare the speed of both techniques

# In[7]:


import requests
url = "https://www.gutenberg.org/files/11/11-0.txt"
alice = requests.get(url)
print(alice.text)


# In[8]:


wordnet_lemmatizer = WordNetLemmatizer()
wordnet_lemmatizer.lemmatize("having")


# In[9]:


get_ipython().run_cell_magic('time', '', "_ = [wordnet_lemmatizer.lemmatize(token, pos='n') for token in word_tokenize(alice.text)]\n")


# In[10]:


get_ipython().run_cell_magic('time', '', '_ = [stemmer.stem(token) for token in word_tokenize(alice.text)]\n')


# * Lemmatising is faster than stemming in this case because the nltk lemmatiser also takes another argument called the part-of-speech (POS) tag of the input word.
# * The default part-of-speech tag is 'noun'..
# * You will learn more about part-of-speech tagging later in this course.
# * Right now, the stemmer will have more accuracy than the lemmatiser because each word is lemmatised assuming it's a noun. To lemmatise efficiently, you need to pass it's POS tag manually.

# In[11]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

