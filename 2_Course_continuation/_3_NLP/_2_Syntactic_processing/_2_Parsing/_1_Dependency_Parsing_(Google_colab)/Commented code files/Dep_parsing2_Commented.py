#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries
import spacy
from spacy import displacy
import pandas as pd
nlp = spacy.load("en_core_web_sm")


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


# ### How do we impliment the rule `if dep nsubjpass, then passive else not`?

# In[ ]:


# Import matcher 
from spacy.matcher import Matcher


# ### Read more about it [here](https://spacy.io/api/matcher)

# In[ ]:


# Visualise the dependency parse tree of 1st sentence of the passive sentences.


# ### Create a rule with `Matcher`

# In[ ]:


#Create rule with matcher


# In[ ]:





# In[ ]:





# In[ ]:





# ### Create a rule for `passive voice`

# In[ ]:





# In[ ]:





# ### Let's check how this rule works if we use it on a sentence with `active voice`

# In[ ]:





# In[ ]:





# In[ ]:





# ### Now lets make a function that impliments this logic

# In[ ]:





# ### Let's test this function on our small sample of sentences and see how the pipeline will work

# In[ ]:





# In[ ]:





# ### Summary
#  - One can go a long way by observing patterns in linguistic data, you don't always need to know the details of the linguitsics very well.
#  - Once can use the `matcher` object to find if certain linguistic patterns exist in data
