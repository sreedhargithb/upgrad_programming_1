#!/usr/bin/env python
# coding: utf-8

# ## Multinomial and Bernoulli Naive Bayes

# For understanding Multinomial and Bernoulli Naive Bayes, we will start with a small example and understand the end to end process. In another notebook, we will build a full-fledged email spam classifier.
# 
# To start with, let's take a few sentences and classify them in two different classes - *education* or *cinema*. Each sentence will represent one document. In real-world cases, a document be any piece of text such as an email, a news article, a book review, a tweet etc.
# The analysis and the algorithm involved doesn’t depend on the type of document we use.

# The notebook is divided into the following sections:
# 1. Importing and preprocessing data
# 2. Building the model: Multinomial Naive Bayes
# 3. Building the model: Bernoulli Naive Bayes

# ### 1. Importing and Preprocessing Data
# 
# Let us first look at the sentences and their classes. We have kept the training sentences in file example_train.csv. Test sentences have been put in the file example_test.csv.

# In[22]:


import numpy as np
import pandas as pd
import sklearn

# training data
train_docs = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_1_Exam_1/2_Machine_learning/3_Naive_Bayes/Naive-Bayes-main/Naive%20Bayes%20for%20text%20classification/example_train.csv') 
print(train_docs)


# So as you can see there are 5 documents (sentences) , 3 are of "education" class and 2 are of "cinema" class.

# In[23]:


# convert label to a numerical variable
train_docs['Class'] = train_docs.Class.map({'cinema':0, 'education':1})
print(train_docs)


# Let's now split the dataframe into X and y labels.

# In[24]:


# convert the df to a numpy array 
train_array = train_docs.values

# split X and y
X_train = train_array[:,0]
y_train = train_array[:,1]
y_train = y_train.astype('int') # sklearn needs y as integers

print("X_train")
print(X_train)
print("y_train")
print(y_train)


# ### Creating the Bag of Words Representation
# 
# We now have to convert the data into a format which can be used for training the model. We'll use the **bag of words representation** for each sentence (document).
# 
# Imagine breaking X in individual words and putting them all in a bag. Then we pick all the unique words from the bag one by one and make a dictionary of unique words. 
# 
# This is called **vectorization of words**. We have the class ```CountVectorizer()``` in scikit learn to vectorize the words. 
# 

# In[25]:


# create an object of CountVectorizer() class 
from sklearn.feature_extraction.text import CountVectorizer 
# help(CountVectorizer)


# In[26]:


vec = CountVectorizer()


# Here ```vec``` is an object of class ```CountVectorizer()```. This has a method called  ```fit()``` which converts a corpus of documents to a matrix of 'tokens'.

# In[27]:


# fit the vectorizer on training data 
vec.fit(X_train)
print(vec.vocabulary_)


# ```Countvectorizer()``` has converted the documents into a set of unique words alphabetically sorted and indexed.
# 
# 
# **Stop Words**
# 
# We can see a few trivial words such as  'and','is','of', etc. These words don't really make any difference in classyfying a document. These are called **stop words**. So we would like to get rid of them. 
# 
# We can remove them by passing a parameter stop_words='english' while instantiating ```Countvectorizer()``` as follows: 

# In[28]:


# fitting the vectorizer on training data again
# removing the stop words this time
vec = CountVectorizer(stop_words='english')
vec.fit(X_train)
print(vec.vocabulary_)


# Notice that the vocabulary has reduced to 12 from 15. Another way of printing the 'vocabulary' is as follows:

# In[29]:


# printing feature names
print(vec.get_feature_names_out())
print(len(vec.get_feature_names_out()))


# So our final dictionary is made of 12 words (after discarding the stop words). Now, to do classification, we need to represent all the documents with these words (or tokens) as features. 
# 
# Every document will be converted into a *feature vector* representing presence of these words in that document. Let's convert each of our training documents in to a feature vector.

# In[30]:


# another way of representing the features
X_transformed = vec.transform(X_train)
print(X_transformed)


# You can see X_tranformed is a 5 x 12 **sparse matrix**. It has 5 rows for each of our 5 documents and 12 columns each 
# for one word of the dictionary which we just created. Let us print X_transformed.

# In[31]:


print(X_transformed)


# This representation can be understood as follows:
# 
# Consider first 4 rows of the output: (0,2), (0,5), (0,7) and (0,11). It says that the first document (index 0) has 
# 7th , 2nd , 5th and 11th 'word' present in the document, and that they appear only
# once in the document- indicated by the right hand column entry. 
# 
# Similarly, consider the entry (4,4) (third from bottom). It says that the fifth document has the fifth word present twice. Indeed, the 5th word('good') appears twice in the 5th document. 
# 
# In real problems, you often work with large documents and vocabularies, and each document contains only a few words in the vocabulary. So it would be a waste of space to store the vocabulary in a typical dataframe, since most entries would be zero. Also, matrix products, additions etc. are much faster with sparse matrices. That's why we use sparse matrices to store the data.
# 
# 
# Let us convert this sparse matrix into a more easily interpretable array:

# In[32]:


# converting transformed matrix back to an array
# note the high number of zeros
X_transformed.toarray()


# To make the dataset more readable, let us examine the vocabulary and the document-term matrix together in a pandas dataframe. The way to convert a matrix into a dataframe is ```pd.DataFrame(matrix, columns=columns)```.
# 

# In[33]:


# converting matrix to dataframe
pd.DataFrame(X_transformed.toarray(), 
             columns=vec.get_feature_names_out())


# This table shows how many times a particular word occurs in document. In other words, this is a frequency table of the words.

# A corpus of documents can thus be represented by a matrix with one row per document and one column per
# token (e.g. word) occurring in the corpus.

# Let's now import and transform the test data as well.

# In[34]:


# test data
test_docs = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_1_Exam_1/2_Machine_learning/3_Naive_Bayes/Naive-Bayes-main/Naive%20Bayes%20for%20text%20classification/example_test.csv') 
print(test_docs)


# In[35]:


# convert label to a numerical variable
test_docs['Class'] = test_docs.Class.map({'cinema':0, 'education':1})
print(test_docs)


# In[36]:


# convert to numpy array
test_numpy_array = test_docs.values

# split into X and y
X_test = test_numpy_array[:,0]
y_test = test_numpy_array[:,1]

print("X_test")
print(X_test)
print("y_test")
print(y_test)


# In[37]:


# transform the test data
# note that you *never* fit on test data, only on training data
# and only transform the test data
X_test_transformed = vec.transform(X_test)
print(X_test_transformed)


# In[38]:


# convert to non-sparse array
X_test=X_test_transformed.toarray()
print(X_test)


# Let us summarise all we have done till now:
# 
# - ```vect.fit(train)``` learns the vocabulary of the training data
# - ```vect.transform(train)``` uses the fitted vocabulary to build a document-term matrix from the training data
# - ```vect.transform(test)``` uses the fitted vocabulary to build a document-term matrix from the testing data (and ignores tokens it hasn't seen before)

# ### 2. Building the Model: Multinomial Naive Bayes

# In[39]:


# building a multinomial NB model
from sklearn.naive_bayes import MultinomialNB

# instantiate NB class
mnb=MultinomialNB()

# fitting the model on training data
mnb.fit(X_transformed, y_train)

# note that we are using the sparse matrix X_transformed, 
# though you can also use the non-sparse version
# mnb.fit(X_transformed.toarray(), y_train) 

# predicting probabilities of test data
proba = mnb.predict_proba(X_test)


# In[40]:


# probability of each class (test data)
print("probability of test document belonging to class CINEMA" , proba[:,0])
print("probability of test document belonging to class EDUCATION" , proba[:,1])


# ### 3. Building the Model: Bernoulli Naive Bayes

# In[41]:


from sklearn.naive_bayes import BernoulliNB

# instantiating bernoulli NB class
bnb=BernoulliNB()

# fitting the model
bnb.fit(X_transformed, y_train)

# also works
# bnb.fit(X_transformed.toarray(), y_train)

# predicting probability of test data
bnb.predict_proba(X_test)
prob_bnb = bnb.predict_proba(X_test)
print(prob_bnb)


# In the next notebook, we will use Multinomial and Bernoulli Naive Bayes to solve an interesting real problem - classifying SMSes as spam or ham. We'll also see how to decide the optimal cutoff probability and evaluate the model.
# 

# In[42]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

