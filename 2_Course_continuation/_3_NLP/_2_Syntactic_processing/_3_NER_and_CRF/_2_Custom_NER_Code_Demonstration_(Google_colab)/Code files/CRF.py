#!/usr/bin/env python
# coding: utf-8

# For this demo, we will use the [MIT Restaurant Corpus](https://groups.csail.mit.edu/sls/downloads/restaurant/) -- a dataset of transcriptions of spoken utterances about restaurants.
# 
# The dataset has following entity types:
# 
# * 'B-Rating'
# * 'I-Rating',
# * 'B-Amenity',
# * 'I-Amenity',
# * 'B-Location',
# * 'I-Location',
# * 'B-Restaurant_Name',
# * 'I-Restaurant_Name',
# * 'B-Price',
# * 'B-Hours',
# * 'I-Hours',
# * 'B-Dish',
# * 'I-Dish',
# * 'B-Cuisine',
# * 'I-Price',
# * 'I-Cuisine'
# 
# Let us load the dataset and see what are we working with.

# In[1]:


# Define the base URL to keep things clean
BASE_URL="https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_3_NLP/_2_Syntactic_processing/_3_NER_and_CRF/_2_Custom_NER_Code_Demonstration_(Google_colab)/Dataset"

# Download all 4 files
get_ipython().system('wget "$BASE_URL/sent_train"')
get_ipython().system('wget "$BASE_URL/label_train"')
get_ipython().system('wget "$BASE_URL/sent_test"')
get_ipython().system('wget "$BASE_URL/label_test"')


# Read training sentences
with open('sent_train', 'r') as train_sent_file:
    train_sentences = train_sent_file.readlines()

# Read training labels
with open('label_train', 'r') as train_labels_file:
    train_labels = train_labels_file.readlines()

# Read testing sentences
with open('sent_test', 'r') as test_sent_file:
    test_sentences = test_sent_file.readlines()

# Read testing labels
with open('label_test', 'r') as test_labels_file:
    test_labels = test_labels_file.readlines()


# Let us see some example data points.

# In[2]:


# Print the 6th sentence in the test set i.e. index value 5.
print(test_sentences[5])

# Print the labels of this sentence
print(test_labels[5])


# #Defining Features for Custom NER

# First, let us install the required modules.

# In[3]:


# Install pycrf and crfsuit packages using pip command
get_ipython().system('pip install pycrf')
get_ipython().system('pip install sklearn-crfsuite')


# 
# 
# We will now start with computing features for our input sequences.

# We have defined the following features for CRF model building:
# 
# - f1 = input word is in lower case; 
# - f2 = last 3 characters of word;
# - f3 = last 2 characers of word;
# - f4 = 1; if the word is in uppercase, 0 otherwise;
# - f5 = 1; if word is a number; otherwise, 0 
# - f6= 1; if the word starts with a capital letter; otherwise, 0
# 

# In[4]:


#Define a function to get the above defined features for a word.

def getFeaturesForOneWord(sentence, pos):
  word = sentence[pos]

  features = [
    'word.lower=' + word.lower(), # serves as word id
    'word[-3:]=' + word[-3:],     # last three characters
    'word[-2:]=' + word[-2:],     # last two characters
    'word.isupper=%s' % word.isupper(),  # is the word in all uppercase
    'word.isdigit=%s' % word.isdigit(),  # is the word a number
    'words.startsWithCapital=%s' % word[0].isupper() # is the word starting with a capital letter
  ]

  if(pos > 0):
    prev_word = sentence[pos-1]
    features.extend([
    'prev_word.lower=' + prev_word.lower(), 
    'prev_word.isupper=%s' % prev_word.isupper(),
    'prev_word.isdigit=%s' % prev_word.isdigit(),
    'prev_words.startsWithCapital=%s' % prev_word[0].isupper()
  ])
  else:
    features.append('BEG') # feature to track begin of sentence 

  if(pos == len(sentence)-1):
    features.append('END') # feature to track end of sentence

  return features


# #Computing Features 

# Define a function to get features for a sentence using the already defined 'getFeaturesForOneWord' function

# In[5]:


# Define a function to get features for a sentence 
# using the 'getFeaturesForOneWord' function.
def getFeaturesForOneSentence(sentence):
  sentence_list = sentence.split()
  return [getFeaturesForOneWord(sentence_list, pos) for pos in range(len(sentence_list))]


# Define function to get the labels for a sentence.

# In[6]:


# Define a function to get the labels for a sentence.
def getLabelsInListForOneSentence(labels):
  return labels.split()


# Example features for a sentence
# 

# In[7]:


# Apply function 'getFeaturesForOneSentence' to get features on a single sentence which is at index value 5 in train_sentences
example_sentence = train_sentences[5]
print(example_sentence)

features = getFeaturesForOneSentence(example_sentence)
features[2]


# Get the features for sentences of X_train and X_test and get the labels of Y_train and Y_test data.

# In[8]:


X_train = [getFeaturesForOneSentence(sentence) for sentence in train_sentences]
Y_train = [getLabelsInListForOneSentence(labels) for labels in train_labels]

X_test = [getFeaturesForOneSentence(sentence) for sentence in test_sentences]
Y_test = [getLabelsInListForOneSentence(labels) for labels in test_labels]


# #CRF Model Training
# 
#  Now we have all the information we need to train our CRF. Let us see how we can do that.

# In[9]:


import sklearn_crfsuite

from sklearn_crfsuite import metrics


# We create a CRF object and passtraining data to it. The model then "trains" and learns the weights for feature functions.

# In[10]:


# Build the CRF model.
crf = sklearn_crfsuite.CRF(max_iterations=100)
crf.fit(X_train, Y_train)


# #Model Testing and Evaluation 
# The model is trained, let us now see how good it performs on the test data.

# In[11]:


# Calculate the f1 score using the test data
Y_pred = crf.predict(X_test)
metrics.flat_f1_score(Y_test, Y_pred, average='weighted')


# In[12]:


# Print the orginal labels and predicted labels for the sentence  in test data, which is at index value 10.
id = 10
print("Sentence:",test_sentences[id])
print("Orig Labels:", Y_test[id])
print("Pred Labels:", Y_pred[id])


# #Transitions Learned by CRF

# In[13]:


get_ipython().system('wget "https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_3_NLP/_2_Syntactic_processing/_3_NER_and_CRF/_2_Custom_NER_Code_Demonstration_(Google_colab)/Code%20files/util.py" -O util.py')
from util import print_top_likely_transitions
from util import print_top_unlikely_transitions


# In[14]:


print_top_likely_transitions(crf.transition_features_)


# In[15]:


print_top_unlikely_transitions(crf.transition_features_)


# In[16]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

