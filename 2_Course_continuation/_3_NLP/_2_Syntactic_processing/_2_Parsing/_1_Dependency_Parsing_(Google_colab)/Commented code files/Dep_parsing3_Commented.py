#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# Import libraries
import spacy
from spacy import displacy
from spacy.matcher import Matcher
import pandas as pd
nlp = spacy.load("en_core_web_sm")


# ### Lets check our rule on a larger corpus

# In[ ]:


# load the dataset csv file


# In[ ]:


# Print the shape of the dataframe.


# In[ ]:


# Separate out active and passive sentences in arrays.


# ### Create the rule

# In[ ]:





# In[ ]:





# ### Check rule on active voice sentences

# In[ ]:





# ### Check rule on passive voice sentences

# In[ ]:





# ### Let's troubleshoot

# In[ ]:





# In[ ]:





# In[ ]:





# ### Let's visualize their dependency trees

# In[ ]:





# In[ ]:





# [Dependencies](https://universaldependencies.org/docs/en/dep/)

# ### Update our rule
# [Reference](https://spacy.io/usage/rule-based-matching)

# In[ ]:





# In[ ]:





# In[ ]:





# ## Summary
#  - Always test your rules and hueristics on a larger corpus to see the effectiveness of the rules
#  - One can write intricate matching rules using `matcher` object
