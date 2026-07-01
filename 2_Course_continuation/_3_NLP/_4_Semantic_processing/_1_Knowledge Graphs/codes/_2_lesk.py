#!/usr/bin/env python
# coding: utf-8

# ## Word-Sense Disambiguation

# In[1]:


from nltk.corpus import wordnet as wn
from nltk import wsd


# In[2]:


X = 'The die is cast.'
Y = 'Roll the die to get a 6.'
Z = 'What is dead may never die.'


# In[3]:


print(wn.synsets('die'))


# In[4]:


print(wn.synsets('die', pos=wn.NOUN))


# In[5]:


for syn in wn.synsets('die', pos=wn.NOUN):
    print(syn.definition())


# In[6]:


for syn in wn.synsets('die', pos=wn.VERB):
    print(syn.definition())


# ## Word-Sense Disambiguation with Lesk Algorithm

# In[7]:


print(X)
_last_lesk = wsd.lesk(X.split(), 'die')


# In[8]:


_last_lesk.definition() if _last_lesk is not None else None


# In[9]:


print(wsd.lesk(X.split(), 'die', pos=wn.NOUN).definition())


# In[10]:


print(Y)
print(wsd.lesk(Y.split(), 'die').definition())


# In[11]:


print(wsd.lesk(Y.split(), 'die', pos=wn.NOUN).definition())


# In[12]:


print(Z)
print(wsd.lesk(Z.split(), 'die').definition())


# In[13]:


print(wsd.lesk(Z.split(), 'die', pos=wn.VERB).definition())


# ## Automatic POS Tagging + Lesk with spaCy

# In[14]:


get_ipython().system('pip install spacy')


# In[15]:


from spacy.cli import download
from spacy import load
# download('en_core_web_sm')
nlp = load('en_core_web_sm')


# In[16]:


import warnings

POS_MAP = {
    'VERB': wn.VERB,
    'NOUN': wn.NOUN,
    'PROPN': wn.NOUN
}


def lesk(doc, word):
    found = False
    for token in doc:
        if token.text == word:
            word = token
            found = True
            break
    if not found:
        raise ValueError(f'Word \"{word}\" does not appear in the document: {doc.text}.')
    pos = POS_MAP.get(word.pos_, False)
    if not pos:
        warnings.warn(f'POS tag for {word.text} not found in wordnet. Falling back to default Lesk behaviour.')
    args = [c.text for c in doc], word.text
    kwargs = dict(pos=pos)
    return wsd.lesk(*args, **kwargs)


# In[17]:


doc = nlp('Roll the die to get a 6.')


# In[18]:


lesk(doc, 'die')


# In[19]:


print(lesk(doc, 'die').definition())


# In[20]:


print(lesk(nlp('I work at google.'), 'google').definition())


# In[21]:


print(lesk(nlp('I will google it.'), 'google').definition())


# In[22]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:




