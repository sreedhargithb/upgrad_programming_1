#!/usr/bin/env python
# coding: utf-8

# ## Plotting word frequencies

# In[1]:


import requests
from nltk import FreqDist
from nltk.corpus import stopwords
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# Download text of 'Alice in Wonderland' ebook from https://www.gutenberg.org/

# In[2]:


url = "https://www.gutenberg.org/files/11/11-0.txt"
alice = requests.get(url)
print(alice.text)


# Define a function to plot word frequencies

# In[3]:


from collections import Counter
import matplotlib.pyplot as plt

def plot_word_frequency(words, top_n):
    word_freq = Counter(words)
    labels = [element[0] for element in word_freq.most_common(top_n)]
    counts = [element[1] for element in word_freq.most_common(top_n)]

    # Plot the bar plot using Seaborn
    plot = sns.barplot(x=labels, y=counts)
    plot.set_xticklabels(plot.get_xticklabels(), rotation=45)
    plt.show()


# Plot words frequencies present in the gutenberg corpus 

# In[4]:


alice_words = alice.text.split()
plot_word_frequency(alice_words, 15)


# ## Stopwords

# Import stopwords from nltk

# In[5]:


from nltk.corpus import stopwords


# Look at the list of stopwords

# In[6]:


import nltk
nltk.download('stopwords')

print(stopwords.words('english'))


# Let's remove stopwords from the following piece of text.

# In[7]:


sample_text = "the great aim of education is not knowledge but action"


# Break text into words

# In[8]:


sample_words = sample_text.split()
print(sample_words)


# Remove stopwords

# In[9]:


sample_words = [word for word in sample_words if word not in stopwords.words('english')]
print(sample_words)


# Join words back to sentence

# In[10]:


sample_text = " ".join(sample_words)
print(sample_text)


# ## Removing stopwords in the genesis corpus

# In[11]:


no_stops = [word for word in alice_words if word not in stopwords.words("english")]


# In[12]:


plot_word_frequency(no_stops, 10)


# Some other things that can be done
# * Need to change tokens to lower case
# * Need to get rid of punctuations
# 
# All the preprocessing steps will be covered while creating the classifier

# In[13]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

