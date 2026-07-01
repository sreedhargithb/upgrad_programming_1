#!/usr/bin/env python
# coding: utf-8
# ----------------------------------------------------------------------
# Filename : intro-to-wordnet.ipynb
# Author   : Jaidev Deshpande
# Purpose  : Understanding Wordnet functionalities
# Libraries: nltk
# ----------------------------------------------------------------------
# ## [WordNet®](https://wordnet.princeton.edu/) Tutorial
# 
# ### Navigating Wornet Relationships

# In[1]:


get_ipython().system('pip install nltk')


# In[2]:


from nltk import download


# In[3]:


download('wordnet')


# In[4]:


from nltk.corpus import wordnet


# In[5]:


# Synsets

tractor = wordnet.synsets('tractor')
tractor


# In[6]:


# Definitions of senses

[syn.definition() for syn in tractor]


# In[7]:


# Hypernyms: Relation between a concept and its superordinate

tractor = wordnet.synset('tractor.n.01')
tractor.hypernyms()


# In[8]:


self_propelled_vehicle = wordnet.synset('self-propelled_vehicle.n.01')
self_propelled_vehicle.hypernyms()


# In[9]:


# Meronyms: Relation between a part and its whole

wheeled_vehicle = wordnet.synset('wheeled_vehicle.n.01')
wheeled_vehicle.part_meronyms()


# In[10]:


# Hyponyms: Relation between a concept and its subordinate

wheeled_vehicle.hyponyms()


# In[11]:


# Holonyms: Relation between whole and its parts

axle = wordnet.synset('axle.n.01')
axle.part_holonyms()


# In[12]:


self_propelled_vehicle.hyponyms()


# In[13]:


motor_vehicle = wordnet.synset('motor_vehicle.n.01')
motor_vehicle.hyponyms()


# In[14]:


car = wordnet.synset('car.n.01')
car.part_meronyms()


# In[15]:


import datetime, pytz;
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

