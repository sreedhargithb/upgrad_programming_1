# load all necessary libraries
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

pd.set_option('max_colwidth', 100)

documents = ["Gangs of Wasseypur is a great movie.", "The success of a movie depends on the performance of the actors.", "There are no new movies releasing this week."]
print(documents)

import nltk
nltk.download('punkt')
nltk.download('stopwords')

def preprocess(document):
    'changes document to lower case and removes stopwords'

    # change sentence to lower case
    document = document.lower()

    # tokenize into words
    words = word_tokenize(document)

    # remove stop words
    words = [word for word in words if word not in stopwords.words("english")]

    # join words to make sentence
    document = " ".join(words)
    
    return document

documents = [preprocess(document) for document in documents]
print(documents)


vectorizer = CountVectorizer()
bow_model = vectorizer.fit_transform(documents)
print(bow_model)  # returns the rown and column number of cells which have 1 as value

# print the full sparse matrix
print(bow_model.toarray())

print(bow_model.shape)
print(vectorizer.get_feature_names_out())

# load data
spam = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_3_NLP/_1_Lexical_processing/_2_Basic_lexical_processing/_2_SMSSpamCollection.txt", sep = "\t", names=["label", "message"])
print(spam.shape)
print(spam.head())

spam = spam.iloc[0:100,:]
print(spam)

# extract the messages from the dataframe
messages = spam.message
print(messages)

# convert messages into list
messages = [message for message in messages]
print(messages)

# preprocess messages using the preprocess function
messages = [preprocess(message) for message in messages]
print(messages)

# bag of words model
vectorizer = CountVectorizer()
bow_model = vectorizer.fit_transform(messages)

# look at the dataframe
pd.DataFrame(bow_model.toarray(), columns = vectorizer.get_feature_names_out())

print(vectorizer.get_feature_names_out())

print(bow_model.shape)

import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))


