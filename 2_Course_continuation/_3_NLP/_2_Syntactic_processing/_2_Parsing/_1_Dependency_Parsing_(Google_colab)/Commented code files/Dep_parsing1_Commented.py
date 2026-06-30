#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries
import spacy
from spacy import displacy
import pandas as pd
nlp = spacy.load("en_core_web_sm")


# ## How to do a dependency parse?

# In[ ]:


# Define active and passive sentences.
active = ['Hens lay eggs.',
         'Birds build nests.',
         'The batter hit the ball.',
         'The computer transmitted a copy of the manual']
passive = ['Eggs are laid by hens',
           'Nests are built by birds',
           'The ball was hit by the batter',
           'A copy of the manual was transmitted by the computer.']


# In[ ]:


# Get the dependency parsing tags


# ### Visualize this parse

# In[ ]:


# Visualise the parse tree


# To understand what these dependency relationships one can use [this link](https://universaldependencies.org/docs/en/dep/)

# ### Going through the dependency relationships it looks like that one would need to know linguistics and grammar to be able to do analysis. This is not entirely true. Many times being able to find out `patterns` in terms of dependency relationships is enough to perform the task at hand

# In[ ]:


# Visualise the parse tree of all the active sentences.


# In[ ]:


# Visualise the parse tree of all the passive sentences.


# ## Summary:
# - Spacy's dependency parser let's us visualise the relationships
# - When a sentence is in passive voice there is always a presence if `nsubjpass` dependency relation

# In[ ]:




