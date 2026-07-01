#!/usr/bin/env python
# coding: utf-8

# **Checking the Length of SMS**

# In[ ]:





# In[1]:


get_ipython().system('pip install nltk')


# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import nltk

df_sms = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_1_Exam_1/2_Machine_learning/3_Naive_Bayes/Naive-Bayes-main/Naive%20Bayes%20for%20text%20classification/spam.csv',encoding='latin-1')
print(df_sms.head())


# **Dropping the unwanted columns Unnamed:2, Unnamed: 3 and Unnamed:4**

# In[3]:


df_sms = df_sms.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)
df_sms = df_sms.rename(columns={"v1":"label", "v2":"sms"})


# In[4]:


print(df_sms.head())


# In[5]:


#Checking the maximum length of SMS
print (len(df_sms))


# In[6]:


print(df_sms.tail())


# In[7]:


#Number of observations in each label spam and ham
print(df_sms.label.value_counts())


# In[8]:


print(df_sms.describe())


# In[9]:


df_sms['length'] = df_sms['sms'].apply(len)
print(df_sms.head())


# In[10]:


import matplotlib.pyplot as plt
import seaborn as sns

get_ipython().run_line_magic('matplotlib', 'inline')
df_sms['length'].plot(bins=50, kind='hist')


# In[11]:


df_sms.hist(column='length', by='label', bins=50,figsize=(10,4))


# In[12]:


df_sms['label'] = df_sms['label'].map({'ham':0, 'spam':1}).astype(int)
print(df_sms.shape)
print(df_sms.head())


# **Bag of Words Approach**

# 
# What we have here in our data set is a large collection of text data (5,572 rows of data). Most ML algorithms rely on numerical data to be fed into them as input, and email/sms messages are usually text heavy.
# We need a way to represent text data for machine learning algorithm and the bag-of-words model helps us to achieve that task.
# It is a way of extracting features from the text for use in machine learning algorithms.
# In this approach, we use the tokenized words for each observation and find out the frequency of each token.
# Using a process which we will go through now, we can convert a collection of documents to a matrix, with each document being a row and each word(token) being the column, and the corresponding (row,column) values being the frequency of occurrence of each word or token in that document.
# 
# For example:
# 
# Lets say we have 4 documents as follows:
# 
# **['Hello, how are you!',
# 'Win money, win from home.',
# 'Call me now',
# 'Hello, Call you tomorrow?']**
# 
# Our objective here is to convert this set of text to a frequency distribution matrix, as follows:
# <img src="https://image.ibb.co/casG7U/countvectorizer.png" alt="table">

# Here as we can see, the documents are numbered in the rows, and each word is a column name, with the corresponding value being the frequency of that word in the document.
# 
# Lets break this down and see how we can do this conversion using a small set of documents.
# 
# To handle this, we will be using sklearns count vectorizer method which does the following:
# 
# 1.  It tokenizes the string(separates the string into individual words) and gives an integer ID to each token.
# 2. It counts the occurrence of each of those tokens.

# **Implementation of Bag of Words Approach**

# Step 1: Convert all strings to their lower case form.

# In[13]:


documents = ['Hello, how are you!',
             'Win money, win from home.',
             'Call me now.',
             'Hello, Call hello you tomorrow?']

lower_case_documents = []
lower_case_documents = [d.lower() for d in documents]
print(lower_case_documents)


# Step 2: Removing all punctuations

# In[14]:


sans_punctuation_documents = []
import string

for i in lower_case_documents:
    sans_punctuation_documents.append(i.translate(str.maketrans("","", string.punctuation)))

print(sans_punctuation_documents)


# Step 3: Tokenization

# In[15]:


preprocessed_documents = [[w for w in d.split()] for d in sans_punctuation_documents]
print(preprocessed_documents)


# Step 4: Count frequencies

# In[16]:


frequency_list = []
import pprint
from collections import Counter

frequency_list = [Counter(d) for d in preprocessed_documents]
pprint.pprint(frequency_list)


# **Implementing Bag of Words in scikit-learn**

# '''
# Here we will look to create a frequency matrix on a smaller document set to make sure we understand how the 
# document-term matrix generation happens. We have created a sample document set 'documents'.
# '''
# documents = ['Hello, how are you!',
#                 'Win money, win from home.',
#                 'Call me now.',
#                 'Hello, Call hello you tomorrow?']

# In[17]:


from sklearn.feature_extraction.text import CountVectorizer
count_vector = CountVectorizer()


# **Data preprocessing with CountVectorizer()**
# 
# In above step, we implemented a version of the CountVectorizer() method from scratch that entailed cleaning our data first. This cleaning involved converting all of our data to lower case and removing all punctuation marks. CountVectorizer() has certain parameters which take care of these steps for us. They are:
# 
# lowercase = True
# 
# The lowercase parameter has a default value of True which converts all of our text to its lower case form.
# 
# token_pattern = (?u)\\b\\w\\w+\\b
# 
# The token_pattern parameter has a default regular expression value of (?u)\\b\\w\\w+\\b which ignores all punctuation marks and treats them as delimiters, while accepting alphanumeric strings of length greater than or equal to 2, as individual tokens or words.
# 
# stop_words
# 
# The stop_words parameter, if set to english will remove all words from our document set that match a list of English stop words which is defined in scikit-learn. Considering the size of our dataset and the fact that we are dealing with SMS messages and not larger text sources like e-mail, we will not be setting this parameter value.

# In[18]:


from sklearn.feature_extraction.text import CountVectorizer

# Sample documents
documents = ["This is the first document.", "This is the second document.", "And the third one.", "Is this the first document?"]

# Create a CountVectorizer object
count_vector = CountVectorizer()

# Fit the model with the documents
count_vector.fit(documents)

# Get the feature names (corrected method)
feature_names = count_vector.get_feature_names_out()

# Print the feature names
print(feature_names)


# In[19]:


doc_array = count_vector.transform(documents).toarray()
print(doc_array)


# In[20]:


from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd

# Sample documents
documents = ["This is the first document.", "This is the second document.", "And the third one.", "Is this the first document?"]

# Create a CountVectorizer object
count_vector = CountVectorizer()

# Fit the model with the documents and transform them into a document-term matrix
doc_array = count_vector.fit_transform(documents).toarray()

# Get the feature names (corrected method)
feature_names = count_vector.get_feature_names_out()

# Create a frequency matrix DataFrame
frequency_matrix = pd.DataFrame(doc_array, columns=feature_names)

# Display the frequency matrix
print(frequency_matrix)


# In[21]:


from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(df_sms['sms'], 
                                                    df_sms['label'], test_size=0.20, 
                                                    random_state=1)


# In[22]:


# Instantiate the CountVectorizer method
count_vector = CountVectorizer()

# Fit the training data and then return the matrix
training_data = count_vector.fit_transform(X_train)

# Transform testing data and return the matrix. 
testing_data = count_vector.transform(X_test)


# **Implementation of Naive Bayes Machine Learning Algorithm **
# 
# I will use  sklearns **sklearn.naive_bayes** method to make predictions on our dataset.
# 
# Specifically, we will be using the **multinomial Naive Bayes** implementation. This particular classifier is suitable for classification with discrete features (such as in our case, word counts for text classification). It takes in integer word counts as its input. On the other hand **Gaussian Naive Bayes** is better suited for continuous data as it assumes that the input data has a Gaussian(normal) distribution.

# In[23]:


from sklearn.naive_bayes import MultinomialNB
import numpy as np

# Ensure that y_train is a numeric type
y_train = np.array(y_train, dtype=int)  # or dtype=float if appropriate

# Create and fit the MultinomialNB model
naive_bayes = MultinomialNB()
naive_bayes.fit(training_data, y_train)


# In[24]:


predictions = naive_bayes.predict(testing_data)


# **Evaluating our model**
# 
# Now that we have made predictions on our test set, our next goal is to evaluate how well our model is doing. There are various mechanisms for doing so, but first let's do quick recap of them.
# 
# **Accuracy** measures how often the classifier makes the correct prediction. It’s the ratio of the number of correct predictions to the total number of predictions (the number of test data points).
# 
# **Precision** tells us what proportion of messages we classified as spam, actually were spam. It is a ratio of true positives(words classified as spam, and which are actually spam) to all positives(all words classified as spam, irrespective of whether that was the correct classification), in other words it is the ratio of
# 
# **[True Positives/(True Positives + False Positives)]**
# 
# **Recall(sensitivity)** tells us what proportion of messages that actually were spam were classified by us as spam. It is a ratio of true positives(words classified as spam, and which are actually spam) to all the words that were actually spam, in other words it is the ratio of
# 
# **[True Positives/(True Positives + False Negatives)]**
# 
# For classification problems that are skewed in their classification distributions like in our case, for example if we had a 100 text messages and only 2 were spam and the rest 98 weren't, accuracy by itself is not a very good metric. We could classify 90 messages as not spam(including the 2 that were spam but we classify them as not spam, hence they would be false negatives) and 10 as spam(all 10 false positives) and still get a reasonably good accuracy score. For such cases, precision and recall come in very handy. These two metrics can be combined to get the F1 score, which is weighted average of the precision and recall scores. This score can range from 0 to 1, with 1 being the best possible F1 score.
# 
# We will be using all 4 metrics to make sure our model does well. For all 4 metrics whose values can range from 0 to 1, having a score as close to 1 as possible is a good indicator of how well our model is doing.

# In[25]:


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
print('Accuracy score: {}'.format(accuracy_score(y_test.to_numpy().astype(int), predictions)))
print('Precision score: {}'.format(precision_score(y_test.to_numpy().astype(int), predictions)))
print('Recall score: {}'.format(recall_score(y_test.to_numpy().astype(int), predictions)))
print('F1 score: {}'.format(f1_score(y_test.to_numpy().astype(int), predictions)))


# One of the major advantages that **Naive Bayes** has over other classification algorithms is its ability to handle an extremely large number of features. In our case, each word is treated as a feature and there are thousands of different words. Also, it performs well even with the presence of irrelevant features and is relatively unaffected by them.
# 
# The other major advantage it has is its relative simplicity. Naive Bayes' works well right out of the box and tuning it's parameters is rarely ever necessary, except usually in cases where the distribution of the data is known. 
# 
# It rarely ever overfits the data.
# 
# Another important advantage is that its model training and prediction times are very fast for the amount of data it can handle. 

# In[26]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

