#!/usr/bin/env python
# coding: utf-8

# In[18]:


get_ipython().system('python -m spacy download en_core_web_md')
get_ipython().system('python -m spacy download en_core_web_sm')


# <pre>
# 1.
# Consider the sentence: “It was the best of times and it was the worst of times.”
# 
# What is the number of children for the word ‘best’ in this sentence?
# </pre>

# In[19]:


import spacy
from spacy import displacy
nlp = spacy.load("en_core_web_sm")
sent = "It was the best of times and it was the worst of times."
sent1 = "Dole was defeated by Clinton"
doc = nlp(sent1)
displacy.render(doc,style="dep", jupyter = True)


# In[ ]:





# In[20]:


import spacy
from spacy.matcher import Matcher
nlp = spacy.load("en_core_web_sm")
matcher = Matcher(nlp.vocab)
pattern = [{"DEP":{"IN":["nsubj","nsubjpass","csubj","csubjpass"]}}]
print(matcher.add("subject", [pattern]))


# In[21]:


import datetime, pytz;
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:





# In[ ]:





# In[ ]:




