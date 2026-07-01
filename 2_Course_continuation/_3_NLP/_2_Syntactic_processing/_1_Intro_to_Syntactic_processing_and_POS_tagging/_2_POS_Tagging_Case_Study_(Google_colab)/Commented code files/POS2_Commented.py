#!/usr/bin/env python
# coding: utf-8

# ## Importing libraries

# In[ ]:


import pandas as pd
import numpy as np
import os


# In[ ]:


import spacy 
nlp = spacy.load("en_core_web_sm")


# ### Read reviews data

# In[ ]:


# Load the Samsung.txt dataset
con=open("../Dataset/Samsung.txt",'r', encoding="utf-8")
samsung_reviews=con.read()
con.close()


# In[ ]:


print(len(samsung_reviews.split("\n")))


# ### Dataset is a text file where each review is in a new line

# In[ ]:


print(samsung_reviews.split("\n")[0:4])


# ### Will our hypothesis hold on real world data? `Product features---POS_NOUN`

# In[ ]:


review1=samsung_reviews.split("\n")[0]
review1=nlp(review1)


# ### Lets do nlp parse on part of one review in our dataset

# In[ ]:





# #### Real world data is usually messy, observe the words `found` and `used`

# In[ ]:





# In[ ]:





# In[ ]:


## Get most frequent lemma forms of nouns


# #### It seems possible that if we extract all the nouns from the reviews and look at the top 5 most frequent lemmatised noun forms, we will be able to identify `What people are talking about?`

# ### Lets repeat this experiment on a larger set of reviews

# In[ ]:





# ### Lets add some way of keeping track of time

# In[ ]:





# In[ ]:





# ### Did you notice anything? What do you think will be the time taken to process all the reviews?

# In[ ]:





# In[ ]:





# ## Summary
# - POS tag based rule seems to be working well
# - We need to figure out a way to reduce the time taken to process reviews
