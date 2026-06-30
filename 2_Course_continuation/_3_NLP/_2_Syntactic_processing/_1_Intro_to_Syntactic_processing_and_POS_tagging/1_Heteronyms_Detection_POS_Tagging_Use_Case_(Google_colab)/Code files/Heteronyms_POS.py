#!/usr/bin/env python
# coding: utf-8

# Google colab link: https://colab.research.google.com/drive/1vbSIiyLu_4S72aKheyma9KE5iWwi6VU9?usp=share_link
# 
# **Heteronyms** are the words that have same spelling but mean different things when pronounced differently. 
# 
# 
# - Recall the word *lead* from the lectures. It can refer to the metal lead or the act of leadership. The two pronounciations have different meanings.
# 
# - For machine translation systems or text to speech systems, the ability to identify the correct sense of the word is crucial.
# 
# 
# 

# Let us have a look at this example:
# 
# https://translate.google.com/?sl=en&tl=hi&text=She%20wished%20she%20could%20desert%20him%20in%20the%20desert.%0A&op=translate
# 
# Example taken from: http://www-personal.umich.edu/~cellis/heteronym.html
# 

# In[26]:


# Import SpaCy library
import spacy 


# In[27]:


# Load pre-trained SpaCy model for performing basic 
# NLP tasks such as POS tagging, parsing, etc.
model = spacy.load("en_core_web_sm")


# In[28]:


#Use the model to process the input sentence
tokens = model("She wished she could desert him in the desert.")


# In[29]:


# Print the tokens and their respective PoS tags.
for token in tokens:
    print(token.text, "--", token.pos_, "--", token.tag_)


# Note here that in the above example, the two instances of *desert* have different PoS tags and hence, the text to speech system can use this information to generate the correct pronounciation. 
# 
# The above task is a specific example of the larger NLP problem called Word Sense Disambiguation (WSD). For words that have more than one meaning, WSD is the problem of identifying the correct meaning of the word based on the context in which the word is used.
# 
# 

# Note that this technique will not work when the different meanings have the same PoS tags.
# 
# https://translate.google.com/?sl=en&tl=hi&text=The%20bass%20swam%20around%20the%20bass%20drum%20on%20the%20ocean%20floor.&op=translate

# In[30]:


# Let's take a new example.
tokens = model("The bass swam around the bass drum on the ocean floor")
for token in tokens:
    print(token.text, "--", token.pos_, "--", token.tag_)


# In[31]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

