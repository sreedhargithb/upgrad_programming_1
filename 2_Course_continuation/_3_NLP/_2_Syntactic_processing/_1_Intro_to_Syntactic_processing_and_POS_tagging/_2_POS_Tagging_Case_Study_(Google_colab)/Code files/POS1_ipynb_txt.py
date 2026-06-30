#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[34]:


get_ipython().system('pip install spacy')
get_ipython().system('python -m spacy download en_core_web_sm')


# In[35]:


import pandas as pd
import numpy as np
import os


# ### How do we identify product features?
# <img src = "./images/keywords.png">

# In[36]:


sent1 = "I loved the screen on this phone."
sent2 = "The battery life on this phone is great."
sent3 = "The speakers are pathetic."


# ### Lets do a POS parse and see if we can figure out some patterns.

# In[37]:


import spacy 
nlp = spacy.load("en_core_web_sm")


# In[38]:


doc1 = nlp(sent1)
for tok in doc1:
    print(tok.text,tok.pos_)


# In[39]:


doc2 = nlp(sent2)
for tok in doc2:
    print(tok.text,tok.pos_)


# In[40]:


doc3 = nlp(sent3)
for tok in doc3:
    print(tok.text,tok.pos_)


# #### **Product features such as `screen`, `battery`, `speaker` have a POS tag of NOUN**

# ## Summary
# - Product features such as `screen`, `battery` and `speaker` have a POS tag of Noun
# - If we can find the frequency count of all the nouns in our data, then by looking at top-n nouns we can find out what product features people are talking about
# - Check hypothesis on a real world dataset

# In[41]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

