#!/usr/bin/env python
# coding: utf-8

# ![](./images/active.png)

# In[8]:


import spacy
from spacy import displacy
import pandas as pd
nlp = spacy.load("en_core_web_sm")


# ## How to do a dependency parse?

# In[9]:


active = ['Hens lay eggs.',
         'Birds build nests.',
         'The batter hit the ball.',
         'The computer transmitted a copy of the manual']
passive = ['Eggs are laid by hens',
           'Nests are built by birds',
           'The ball was hit by the batter',
           'A copy of the manual was transmitted by the computer.']


# In[10]:


doc = nlp(active[0])
for tok in doc:
    print(tok.text,tok.dep_)


# ### Visualize this parse

# In[11]:


displacy.render(doc, style="dep", jupyter = True)


# To understand what these dependency relationships one can use [this link](https://universaldependencies.org/docs/en/dep/)

# ### Going through the dependency relationships it looks like that one would need to know linguistics and grammar to be able to do analysis. This is not entirely true. Many times being able to find out `patterns` in terms of dependency relationships is enough to perform the task at hand

# In[12]:


for sent in active:
    doc = nlp(sent)
    displacy.render(doc, style="dep")


# In[13]:


for sent in passive:
    doc = nlp(sent)
    displacy.render(doc, style="dep")


# ## Summary:
# - Spacy's dependency parser let's us visualise the relationships
# - When a sentence is in passive voice there is always a presence if `nsubjpass` dependency relation

# In[14]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

