#!/usr/bin/env python
# coding: utf-8

# In[1]:


import spacy
from spacy import displacy
import pandas as pd
nlp = spacy.load("en_core_web_sm")


# In[2]:


active = ['Hens lay eggs.',
         'Birds build nests.',
         'The batter hit the ball.',
         'The computer transmitted a copy of the manual']
passive = ['Eggs are laid by hens',
           'Nests are built by birds',
           'The ball was hit by the batter',
           'A copy of the manual was transmitted by the computer.']


# ### How do we impliment the rule `if dep nsubjpass, then passive else not`?

# In[3]:


from spacy.matcher import Matcher


# ### Read more about it [here](https://spacy.io/api/matcher)

# In[4]:


doc = nlp(passive[0])
displacy.render(doc, style="dep")


# ### Create a rule with `Matcher`

# In[5]:


rule = [{'POS':'NOUN'}]
matcher = Matcher(nlp.vocab)
matcher.add('Rule',[rule])


# In[6]:


matcher(doc)


# In[7]:


doc[0:1]


# In[8]:


doc[4:5]


# ### Create a rule for `passive voice`

# In[9]:


passive_rule = [{'DEP':'nsubjpass'}]
matcher = Matcher(nlp.vocab)
matcher.add('Rule',[passive_rule])


# In[10]:


matcher(doc)


# ### Let's check how this rule works if we use it on a sentence with `active voice`

# In[11]:


active[0]


# In[12]:


doc = nlp(active[0])
displacy.render(doc, style="dep")


# In[13]:


matcher(doc)


# ### Now lets make a function that impliments this logic

# In[14]:


def is_passive(doc,matcher):
    if len(matcher(doc))>0:
        return True
    else:
        return False


# ### Let's test this function on our small sample of sentences and see how the pipeline will work

# In[15]:


for sent in active:
    doc = nlp(sent)
    print(is_passive(doc,matcher))


# In[16]:


for sent in passive:
    doc = nlp(sent)
    print(is_passive(doc,matcher))


# ### Summary
#  - One can go a long way by observing patterns in linguistic data, you don't always need to know the details of the linguitsics very well.
#  - Once can use the `matcher` object to find if certain linguistic patterns exist in data

# In[17]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:




