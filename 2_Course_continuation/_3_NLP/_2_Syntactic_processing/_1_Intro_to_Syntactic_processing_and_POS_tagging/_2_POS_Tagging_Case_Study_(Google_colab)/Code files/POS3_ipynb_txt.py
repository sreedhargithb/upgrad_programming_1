#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[17]:


import pandas as pd
import numpy as np
import os
import spacy 
from tqdm import tqdm


# ### Read reviews data

# In[18]:


# 1. Download the file from the URL to the Colab disk
get_ipython().system('wget "https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_3_NLP/_2_Syntactic_processing/_1_Intro_to_Syntactic_processing_and_POS_tagging/_2_POS_Tagging_Case_Study_(Google_colab)/Dataset/Samsung.txt"')

# 2. Open the local file (matching your original code)
con = open("Samsung.txt", 'r', encoding="utf-8")
samsung_reviews = con.read()
con.close()


# ### Can we reduce the time taken?
# [Pipelines (Spacy)](https://spacy.io/usage/processing-pipelines)
# 

# <img src='./images/spacy_pipeline.png'>

# In[19]:


# shorten the pipline loading
nlp=spacy.load('en_core_web_sm',disable=['parser','ner'])


# In[20]:


nouns = []
for review in tqdm(samsung_reviews.split("\n")[0:1000]):
    doc = nlp(review)
    for tok in doc:
        if tok.pos_=="NOUN":
            nouns.append(tok.lemma_.lower())


# In[21]:


print(len(samsung_reviews.split("\n")))


# In[22]:


(46355/1000)*6


# In[23]:


278/60


# ### Lets process all the reviews now and see if time taken is less !!!

# In[24]:


nouns = []
for review in tqdm(samsung_reviews.split("\n")):
    doc = nlp(review)
    for tok in doc:
        if tok.pos_=="NOUN":
            nouns.append(tok.lemma_.lower())


# ### Does the hypothesis of nouns capturing `product features` hold?

# In[25]:


nouns=pd.Series(nouns)
print(nouns.value_counts().head(5))


# In[26]:


print(nouns.value_counts().head(10))


# ### We now know that people mention `battery`, `product`, `screen` etc. But we still don't know in what context they mention these keywords

# ### Summary:
#  - Most frequently used lemmatised forms of noun, inform us about the product features people are talking about in product reviews
#  - In order to process the review data faster spacy allows us to use the idea of enabling parts of model inference pipeline via `spacy.loads()` command and `disable` parameter

# In[27]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

