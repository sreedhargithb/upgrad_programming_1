#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import os
import spacy 
from tqdm import tqdm


# ### Read reviews data

# In[ ]:


con=open("../data/Samsung.txt",'r', encoding="utf-8")
samsung_reviews=con.read()
con.close()


# ### Can we reduce the time taken?
# [Pipelines (Spacy)](https://spacy.io/usage/processing-pipelines)
# 

# <img src='./images/spacy_pipeline.png'>

# In[ ]:


# shorten the pipline loading


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Lets process all the reviews now and see if time taken is less !!!

# In[ ]:





# ### Does the hypothesis of nouns capturing `product features` hold?

# In[ ]:





# In[ ]:





# ### We now know that people mention `battery`, `product`, `screen` etc. But we still don't know in what context they mention these keywords

# ### Summary:
#  - Most frequently used lemmatised forms of noun, inform us about the product features people are talking about in product reviews
#  - In order to process the review data faster spacy allows us to use the idea of enabling parts of model inference pipeline via `spacy.loads()` command and `disable` parameter
