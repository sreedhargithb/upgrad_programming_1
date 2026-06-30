#!/usr/bin/env python
# coding: utf-8

# # Tokenisation
# 
# The notebook contains three types of tokenisation techniques:
# 1. Word tokenisation
# 2. Sentence tokenisation
# 3. Tweet tokenisation
# 4. Custom tokenisation using regular expressions

# ### 1. Word tokenisation

# In[8]:


document = "At nine o'clock I visited him myself. It looks like religious mania, and he'll soon think that he himself is God."
print(document)


# Tokenising on spaces using python

# In[9]:


print(document.split())


# Tokenising using nltk word tokeniser

# In[10]:


from nltk.tokenize import word_tokenize
import nltk
nltk.download('punkt_tab')
words = word_tokenize(document)


# In[11]:


print(words)


# NLTK's word tokeniser not only breaks on whitespaces but also breaks contraction words such as he'll into "he" and "'ll". On the other hand it doesn't break "o'clock" and treats it as a separate token.

# ### 2. Sentence tokeniser

# Tokenising based on sentence requires you to split on the period ('.'). Let's use nltk sentence tokeniser.

# In[12]:


from nltk.tokenize import sent_tokenize
sentences = sent_tokenize(document)


# In[13]:


print(sentences)


# ### 3. Tweet tokeniser

# A problem with word tokeniser is that it fails to tokeniser emojis and other complex special characters such as word with hashtags. Emojis are common these days and people use them all the time.

# In[14]:


message = "i recently watched this show called mindhunters:). i totally loved it 😍. it was gr8 <3. #bingewatching #nothingtodo 😎"


# In[15]:


print(word_tokenize(message))


# The word tokeniser breaks the emoji '<3' into '<' and '3' which is something that we don't want. Emojis have their own significance in areas like sentiment analysis where a happy face and sad face can salone prove to be a really good predictor of the sentiment. Similarly, the hashtags are broken into two tokens. A hashtag is used for searching specific topics or photos in social media apps such as Instagram and facebook. So there, you want to use the hashtag as is.
# 
# Let's use the tweet tokeniser of nltk to tokenise this message.

# In[16]:


from nltk.tokenize import TweetTokenizer
tknzr = TweetTokenizer()


# In[17]:


tknzr.tokenize(message)


# As you can see, it handles all the emojis and the hashtags pretty well.
# 
# Now, there is a tokeniser that takes a regular expression and tokenises and returns result based on the pattern of regular expression.
# 
# Let's look at how you can use regular expression tokeniser.

# In[18]:


from nltk.tokenize import regexp_tokenize
message = "i recently watched this show called mindhunters:). i totally loved it 😍. it was gr8 <3. #bingewatching #nothingtodo 😎"
pattern = "#[\w]+"


# In[19]:


regexp_tokenize(message, pattern)


# In[20]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

