#!/usr/bin/env python
# coding: utf-8

# In[16]:


import re
from collections import Counter


# In[17]:


# function to tokenise words
def words(document):
    "Convert text to lower case and tokenise the document"
    return re.findall(r'\w+', document.lower())


# In[18]:


# Run this in a cell first to download the file locally
get_ipython().system('wget https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_3_NLP/_1_Lexical_processing/_3_Advanced_lexical_processing/_3_seed_document.txt -O _3_seed_document.txt')

# all_words = Counter(open('_3_seed_document.txt').read().split())
all_words = Counter(words(open('_3_seed_document.txt').read()))


# In[19]:


# check frequency of a random word, say, 'chair'
print(all_words['chair'])


# In[20]:


# look at top 10 frequent words
all_words.most_common(10)


# In[21]:


def edits_one(word):
    "Create all edits that are one edit away from `word`."
    alphabets    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])                   for i in range(len(word) + 1)]
    deletes    = [left + right[1:]                       for left, right in splits if right]
    inserts    = [left + c + right                       for left, right in splits for c in alphabets]
    replaces   = [left + c + right[1:]                   for left, right in splits if right for c in alphabets]
    transposes = [left + right[1] + right[0] + right[2:] for left, right in splits if len(right)>1]
    return set(deletes + inserts + replaces + transposes)


# In[22]:


def edits_two(word):
    "Create all edits that are two edits away from `word`."
    return (e2 for e1 in edits_one(word) for e2 in edits_one(e1))


# In[23]:


def known(words):
    "The subset of `words` that appear in the `all_words`."
    return set(word for word in words if word in all_words)


# In[24]:


def possible_corrections(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits_one(word)) or known(edits_two(word)) or [word])


# In[25]:


def prob(word, N=sum(all_words.values())): 
    "Probability of `word`: Number of appearances of 'word' / total number of tokens"
    return all_words[word] / N


# In[26]:


print(len(set(edits_one("monney"))))
print(edits_one("monney"))


# In[27]:


print(known(edits_one("monney")))


# In[28]:


# Let's look at words that are two edits away
print(len(set(edits_two("monney"))))
print(known(edits_one("monney")))


# In[29]:


# Let's look at possible corrections of a word
print(possible_corrections("monney"))


# In[30]:


# Let's look at probability of a word
print(prob("money"))
print(prob("monkey"))


# In[31]:


def spell_check(word):
    "Print the most probable spelling correction for `word` out of all the `possible_corrections`"
    correct_word = max(possible_corrections(word), key=prob)
    if correct_word != word:
        return "Did you mean " + correct_word + "?"
    else:
        return "Correct spelling."


# In[32]:


# test spell check
print(spell_check("monney"))


# In[33]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

