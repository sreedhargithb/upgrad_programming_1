#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
from tqdm import tqdm


# ### Read reviews data

# In[ ]:


con=open("../Dataset/Samsung.txt",'r', encoding="utf-8")
samsung_reviews=con.read()
con.close()


# <img src = "./images/results.png">

# <img src = "./images/keywords.png">

# ### We can use a simple hueristic
#  - Find out what were the most common words that appeared before and after each mention of `product feature`
#  - Use regex pattern to extract this information

# In[ ]:





# The `battery` was ===> Prefix `keyword` Suffix

# ![image.png](attachment:9b4e9e8f-7d79-4d31-b370-44726e017a96.png)<img src="./images/regex.png">

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# #### Extract all the prefixes and suffixes of `battery`

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### This doesn't make much sense as these are commonly used words. Let's remove `stopwords` and see what we get
# 
# <a href = "https://gist.github.com/sebleier/554280">Get Stop Words</a>

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# ### Lets pretty print

# In[ ]:





# In[ ]:





# ### Lets put all this logic in a function

# In[ ]:





# In[ ]:





# In[ ]:





# ## Summary:
#     - Simple hueristics sometime are very usefull
#     - Regex can be life saviours
#     - Don't forget to use simple text processing while trying to solve a non-trival problem

# In[ ]:




