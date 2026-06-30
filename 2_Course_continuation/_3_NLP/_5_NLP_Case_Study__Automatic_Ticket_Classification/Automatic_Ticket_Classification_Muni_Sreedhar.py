#!/usr/bin/env python
# coding: utf-8

# # NLP Case Study - Automatic Ticket Classification
# ##### By Sreedhar K and Munirathinam Duraisamy

# ## Contents:
# 
# - [Problem Statement](#Problem_Statement)<br>
# 
# - [Pipelines that need to be performed](#Pipelines_that_need_to_be_performed)<br>
# 
# - [Importing the necessary libraries](#Importing_the_necessary_libraries)<br>

# <a id="Problem_Statement"></a>
# ## Problem Statement
# 
# You need to build a model that is able to classify customer complaints based on the products/services. By doing so, you can segregate these tickets into their relevant categories and, therefore, help in the quick resolution of the issue.
# 
# You will be doing topic modelling on the <b>.json</b> data provided by the company. Since this data is not labelled, you need to apply NMF to analyse patterns and classify tickets into the following five clusters based on their products/services:
# 
# * Credit card / Prepaid card
# 
# * Bank account services
# 
# * Theft/Dispute reporting
# 
# * Mortgages/loans
# 
# * Others
# 
# 
# With the help of topic modelling, you will be able to map each ticket onto its respective department/category. You can then use this data to train any supervised model such as logistic regression, decision tree or random forest. Using this trained model, you can classify any new customer complaint support ticket into its relevant department.

# <a id="Pipelines_that_need_to_be_performed"></a>
# ## Pipelines that need to be performed:
# 
# You need to perform the following eight major tasks to complete the assignment:
# 
# 1.  Data loading
# 
# 2. Text preprocessing
# 
# 3. Exploratory data analysis (EDA)
# 
# 4. Feature extraction
# 
# 5. Topic modelling
# 
# 6. Model building using supervised learning
# 
# 7. Model training and evaluation
# 
# 8. Model inference

# <a id="Importing_the_necessary_libraries"></a>
# ## Importing the necessary libraries

# In[1]:


get_ipython().system('pip install textblob swifter')


# In[2]:


import json
import numpy as np
import pandas as pd
import re, nltk, spacy, string
import en_core_web_sm
nlp = en_core_web_sm.load()
import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')

from plotly.offline import plot
import plotly.graph_objects as go
import plotly.express as px

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from pprint import pprint
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfTransformer
from IPython.display import HTML, display


# <a id="Loading_the_data"></a>
# ## Loading the data
# 
# The data is in JSON format and we need to convert it to a dataframe.

# In[3]:


# Opening JSON file
get_ipython().system('wget https://raw.githubusercontent.com/VJ-Jain/NLP-Automatic-Ticket-Classification/refs/heads/master/complaints-2021-05-14_08_16.json')
f = open('complaints-2021-05-14_08_16.json')

# returns JSON object as
# a dictionary
data = json.load(f)
df=pd.json_normalize(data)


# <a id="Data_preparation"></a>
# ## Data preparation

# In[4]:


# Inspect the dataframe to understand the given data.
display(
    df.head()
)




# In[5]:


#print the column names
display(
    df.shape
)




# In[6]:


#print the column names
col_list=list(df.columns)
display(
    col_list
)



# In[7]:


display(
    df.info()
)


# In[8]:


display(
    df.describe()
)


# In[9]:


#Find the % of null values in each columns
display(
    round(df.isna().sum()*100/78313,2)
)



# In[10]:


#Assign new column names
#Removing "_" and "source." from column names
df.columns=[re.sub('^_','',col) for col in df.columns]

df.columns = [re.sub(r"^\bsource\b\.", "", col) for col in df.columns]

display(
    list(df.columns)
)



# In[11]:


#Assign nan in place of blanks in the complaints column
df['complaint_what_happened'].replace("", np.nan, inplace=True)



# In[12]:


#Remove all rows where complaints column is nan
#Dropping NaN rows from "_source.complaint_what_happened"
df.dropna(subset=['complaint_what_happened'], inplace=True)


# In[13]:


#New shape of df
display(
    df.shape
)


# <a id="Prepare_the_text_for_topic_modeling"></a>
# ## Prepare the text for topic modeling
# 
# Once you have removed all the blank complaints, you need to:
# 
# * Make the text lowercase
# * Remove text in square brackets
# * Remove punctuation
# * Remove words containing numbers
# 
# 
# Once you have done these cleaning operations you need to perform the following:
# * Lemmatize the texts
# * Extract the POS tags of the lemmatized text and remove all the words which have tags other than NN[tag == "NN"].
# 

# In[14]:


# Write your function here to clean the text and remove all the unnecessary elements.
def clean_texts(text):
    text = text.lower() # to lower case
    text = text.strip() # trim operation
    text = re.sub(r"[\[].*?[\]]","",text).strip() #remove within square bracket
    text = re.sub(r"[^\w\s]","",text).strip() # remove punctuation
    text = re.sub("\S*\d\S*","",text).strip() # remove words containing numbers
    return text


# In[15]:


#Cleaning df['complaint_what_happened']
df['complaint_what_happened']= df['complaint_what_happened'].apply(lambda x: clean_texts(x))


# In[16]:


# Import the necessary libraries for cleanup
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')


#!pip install swifter


# In[17]:


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import swifter #


# In[18]:


#Write your function to Lemmatize the texts
stopwords = stopwords.words('english')
wordnet_lem = WordNetLemmatizer()

def lemma_texts(text):
    tokens = word_tokenize(text)
    tokens = [word for word in tokens if word not in stopwords]
    proc_token = []
    for word,pos in pos_tag(tokens):
        try:
            proc_token.append(wordnet_lem.lemmatize(word,pos = pos[0].lower()))
        except:
            proc_token.append(wordnet_lem.lemmatize(word))
    lemmatized_value = ' '.join(proc_token)
    return lemmatized_value


# In[19]:


import nltk
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

# Add a column for lemmatized complaints to the dataframe
df["lemmatized_complaint"] =  df["complaint_what_happened"].swifter.apply(lemma_texts)

# View the dataframe
display(
    df.head()
)


# In[20]:


#Create a dataframe('df_clean') that will have only the complaints and the lemmatized complaints
df_clean=df[['complaint_what_happened','lemmatized_complaint']]



# In[21]:


display(
    df_clean
)


# In[22]:


#Write your function to extract the POS tags

def pos_tag(text):
  # write your code here

 # Creating a textblob object
    text_blob = TextBlob(text)

    # extracting words with tags 'NN', joining them and return
    return ' '.join([ word for (word,tag) in text_blob.tags if tag == "NN"])


df_clean["complaint_POS_removed"] =  df_clean.apply(lambda x: pos_tag(x['lemmatized_complaint']), axis=1)
 #this column should contain lemmatized text with all the words removed which have tags other than NN[tag == "NN"].
#Reassigning index of cleaned dataframe
df_clean.reset_index(drop=True, inplace = True)

# View the dataframe
display(
    df_clean.head()
)



# In[23]:


#The clean dataframe should now contain the raw complaint, lemmatized complaint and the complaint after removing POS tags.
display(
    df_clean
)


# <a id="EDA"></a>
# ## Exploratory data analysis to get familiar with the data.
# 
# Write the code in this task to perform the following:
# 
# *   Visualise the data according to the 'Complaint' character length
# *   Using a word cloud find the top 40 words by frequency among all the articles after processing the text
# *   Find the top unigrams,bigrams and trigrams by frequency among all the complaints after processing the text. ‘
# 
# 
# 

# In[24]:


#length of character in 'complaint_POS_removed'
char_len=[len(x) for x in df_clean['complaint_POS_removed']]
display(
    char_len[:10]
)


# In[25]:


# Write your code here to visualise the data according to the 'Complaint' character length
bins = [0, 100, 500, 1000, 5000, 10000, 50000]
temp_df = df_clean['complaint_what_happened'].str.len().to_frame()
temp_df.columns = ["length"]
temp_df['binned'] = pd.cut(temp_df['length'], bins)
temp_df.binned.value_counts()

plt.figure(figsize=(12,8))
temp_df.binned.value_counts().plot(kind="bar")
plt.title("Complaint count by character length")
plt.xlabel("Complaint Character length ranges")


# #### Find the top 40 words by frequency among all the articles after processing the text.

# In[26]:


# Most frequent words in the processed (lemmatized) complaints

most_freq_lem=[]
for complaint in df_clean['lemmatized_complaint']:
    for word in complaint.split(' '):
        most_freq_lem.append(word)

plt.figure(figsize=(20, 5))
pd.DataFrame(most_freq_lem)[0].value_counts().head(40).plot(kind='bar')
plt.title("Most frequent words in complaints after processing")


# In[27]:


get_ipython().system('pip install wordcloud')


# In[28]:


#Using a word cloud find the top 40 words by frequency among all the articles after processing the text
from wordcloud import WordCloud, STOPWORDS
stop_words = set(STOPWORDS)
word_cloud = WordCloud(
                          background_color='blue',
                          stopwords=stop_words,
                          max_font_size=38,
                          max_words=38,
                          random_state=42
                         ).generate(str(df_clean['complaint_POS_removed']))

fig = plt.figure(figsize=(20,16))
plt.imshow(word_cloud)
plt.axis('off')
plt.show()


# In[29]:


# Most frequent nouns in the processed complaints
most_freq_nouns=[]
for complaint in df_clean['complaint_POS_removed']:
    for word in complaint.split(' '):
        most_freq_nouns.append(word)

plt.figure(figsize=(20, 5))
pd.DataFrame(most_freq_nouns)[0].value_counts().head(40).plot(kind='bar')
plt.title("Most frequent nouns in complaint dataset")
plt.show()


# In[30]:


#Removing -PRON- from the text corpus
df_clean['complaint_clean'] = df_clean['complaint_POS_removed'].str.replace('-PRON-', '')
display(
    df_clean
)


# #### Find the top unigrams,bigrams and trigrams by frequency among all the complaints after processing the text.

# In[31]:


#Write your code here to find the top 30 unigram frequency among the complaints in the cleaned datafram(df_clean).
def get_top_unigram(text, n=30):

    vector = CountVectorizer(stop_words='english').fit(text)
    bag_of_words = vector.transform(text)
    sum_of_words = bag_of_words.sum(axis=0)
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    word_freq =sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[:n]


# In[32]:


#Print the top 10 words in the unigram frequency
top_common_words = get_top_unigram(df_clean['complaint_POS_removed'].values.astype('U'))
df_unigram = pd.DataFrame(top_common_words, columns = ['unigram' , 'count'])
display(
    df_unigram.head(10)
)


# In[33]:


# Top 30 words by their frequency
plt.figure(figsize=(15,6))
sns.barplot(x='unigram', y='count', data=df_unigram, palette="rocket")
plt.xticks(rotation=90)
plt.title("Top 30 unigrams in the Complaint text after removing stop words and lemmatization", fontsize=20)
plt.show()


# In[34]:


#Write your code here to find the top 30 bigram frequency among the complaints in the cleaned datafram(df_clean).
def get_top_bigram(text, n=30):

    vector = CountVectorizer(ngram_range=(2, 2), stop_words='english').fit(text)
    bag_of_words = vector.transform(text)
    sum_of_words = bag_of_words.sum(axis=0)
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    word_freq =sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[:n]


# In[35]:


#Print the top 10 words in the bigram frequency
top_common_words = get_top_bigram(df_clean['complaint_POS_removed'].values.astype('U'))
df_bigram = pd.DataFrame(top_common_words, columns = ['bigram' , 'count'])
display(
    df_bigram.head(10)
)


# In[36]:


# Plot the top 30 bigrams
plt.figure(figsize=(15,6))
sns.barplot(x='bigram', y='count', data=df_bigram, palette="flare")
plt.xticks(rotation=90)
plt.title("Top 30 bigrams in the Complaint text after removing stop words and lemmatization", fontsize=20)
plt.show()


# In[37]:


#Write your code here to find the top 30 trigram frequency among the complaints in the cleaned datafram(df_clean).
def get_top_trigram(text, n=30):

    vector = CountVectorizer(ngram_range=(3, 3), stop_words='english').fit(text)
    bag_of_words = vector.transform(text)
    sum_of_words = bag_of_words.sum(axis=0)
    word_freq = [(word, sum_of_words[0, idx]) for word, idx in vector.vocabulary_.items()]
    word_freq =sorted(word_freq, key = lambda x: x[1], reverse=True)
    return word_freq[:n]


# In[38]:


#Print the top 10 words in the trigram frequency
top_common_words = get_top_trigram(df_clean['complaint_POS_removed'].values.astype('U'))
df_trigram = pd.DataFrame(top_common_words, columns = ['trigram' , 'count'])
display(
    df_trigram.head(10)
)


# In[39]:


# Plot the top 30 unigrams
plt.figure(figsize=(15,6))
sns.barplot(x='trigram', y='count', data=df_trigram, palette="crest")
plt.xticks(rotation=90)
plt.title("Top 30 trigrams in the Complaint text after removing stop words and lemmatization", fontsize=20)
plt.show()


# ## The personal details of customer has been masked in the dataset with xxxx. Let's remove the masked text as this will be of no use for our analysis

# In[40]:


df_clean['complaint_clean'] = df_clean['complaint_clean'].str.replace('xxxx','')
df_clean['complaint_POS_removed'] = df_clean['complaint_POS_removed'].str.replace('xxxx','')


# In[41]:


#All masked texts has been removed
display(
    df_clean
)


# <a id="Feature_Extraction"></a>
# ## Feature Extraction
# Convert the raw texts to a matrix of TF-IDF features
# 
# **max_df** is used for removing terms that appear too frequently, also known as "corpus-specific stop words"
# max_df = 0.95 means "ignore terms that appear in more than 95% of the complaints"
# 
# **min_df** is used for removing terms that appear too infrequently
# min_df = 2 means "ignore terms that appear in less than 2 complaints"

# In[42]:


# Taking a copy of df_clean
df_cleaner=df_clean.copy()
#df_clean=df_cleaner.copy()


# In[43]:


display(
    df_cleaner.shape
)


# In[44]:


display(
    df_clean.shape
)


# In[45]:


#Write your code here to initialise the TfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(
    min_df=2,
    max_df=0.95,
    stop_words='english',
    max_features=5000,
    ngram_range=(1, 3)
)



# #### Create a document term matrix using fit_transform
# 
# The contents of a document term matrix are tuples of (complaint_id,token_id) tf-idf score:
# The tuples that are not there have a tf-idf score of 0

# In[46]:


#Write your code here to create the Document Term Matrix by transforming the complaints column present in df_clean.
tfidf = tfidf_vectorizer.fit_transform(df_clean['complaint_POS_removed'])
features = tfidf_vectorizer.get_feature_names_out()

# Creating a dataframe to display the document-term matrix created above
tfidf_df = pd.DataFrame(tfidf.toarray(), columns = features)
display(
    tfidf_df.head()
)


# In[47]:


# Check the shape of dataframe
display(
    tfidf_df.shape
)


# <a id="Topic_Modelling"></a>
# ## Topic Modelling
# 
# ### Topic Modelling using NMF
# 
# Non-Negative Matrix Factorization (NMF) is an unsupervised technique so there are no labeling of topics that the model will be trained on. The way it works is that, NMF decomposes (or factorizes) high-dimensional vectors into a lower-dimensional representation. These lower-dimensional vectors are non-negative which also means their coefficients are non-negative.
# 
# In this task you have to perform the following:
# 
# * Find the best number of clusters
# * Apply the best number to create word clusters
# * Inspect & validate the correction of each cluster wrt the complaints
# * Correct the labels if needed
# * Map the clusters to topics/cluster names

# In[48]:


from sklearn.decomposition import NMF


# ### Manual Topic Modeling
# You need to do take the trial & error approach to find the best num of topics for your NMF model.
# 
# The only parameter that is required is the number of components i.e. the number of topics we want. This is the most crucial step in the whole topic modeling process and will greatly affect how good your final topics are.

# In[49]:


#Load your nmf_model with the n_components i.e 5
num_topics = 5 #write the value you want to test out

#keep the random_state =40
nmf_model = NMF(n_components=num_topics, random_state=40)
display(
    len(tfidf_vectorizer.get_feature_names_out())
)


# In[50]:


W = nmf_model.fit_transform(tfidf)  # Document-Topic matrix
H = nmf_model.components_


# In[51]:


#Print the Top15 words for each of the topics
words = np.array(tfidf_vectorizer.get_feature_names_out())
topic_words = pd.DataFrame(np.zeros((num_topics, 15)), index=[f'Topic {i + 1}' for i in range(num_topics)],
                           columns=[f'Word {i + 1}' for i in range(15)]).astype(str)
for i in range(num_topics):
    ix = H[i].argsort()[::-1][:15]
    topic_words.iloc[i] = words[ix]

display(topic_words)


# In[52]:


#Create the best topic for each complaint in terms of integer value 0,1,2,3 & 4
topic_mapping = {
    'Topic 1' : 0,
    'Topic 2' : 1,
    'Topic 3' : 2,
    'Topic 4' : 3,
    'Topic 5' : 4
}



# In[53]:


W = pd.DataFrame(W, columns=[f'Topic {i+1}' for i in range(num_topics)])
W['max_topic'] = W.apply(lambda x: topic_mapping.get(x.idxmax()) if x.idxmax() in topic_mapping.keys() else '4', axis=1)
display(
    W[pd.notnull(W['max_topic'])].head(10)
)


# In[54]:


# Checking the frequency
display(
    W['max_topic'].value_counts()
)


# In[55]:


display(
    df_clean.shape
)


# In[56]:


#Assign the best topic to each of the cmplaints in Topic Column

df_clean['Topic'] =  W['max_topic'].apply(lambda x:int(x)) #write your code to assign topics to each rows.


# In[57]:


display(df_clean.shape)


# In[58]:


display(df_clean.head())


# In[59]:


display(df_clean.shape)


# In[60]:


# Checking the frequency
display(W['max_topic'].value_counts())


# In[61]:


#Print the first 5 Complaint for each of the Topics
df_clean_topic=df_clean.groupby('Topic').head(5)
display(
    df_clean_topic.sort_values('Topic')
)


# #### After evaluating the mapping, if the topics assigned are correct then assign these names to the relevant topic:
# * Bank Account services
# * Credit card or prepaid card
# * Theft/Dispute Reporting
# * Mortgage/Loan
# * Others

# In[62]:


#Create the dictionary of Topic names and Topics

Topic_names = {    0 : "Bank account services",
                1 : "Credit Card/Prepaid Card",
                2 : "Mortgages/loans",
                3 : "Theft/Dispute reporting",
                4 : "Others" }
#Replace Topics with Topic Names
df_clean['Topic'] = df_clean['Topic'].map(Topic_names)


# In[63]:


display(df_clean)


# In[64]:


display(df_clean.shape)


# ## Supervised model to predict any new complaints to the relevant Topics.
# 
# You have now build the model to create the topics for each complaints.Now in the below section you will use them to classify any new complaints.
# 
# Since you will be using supervised learning technique we have to convert the topic names to numbers(numpy arrays only understand numbers)

# In[65]:


#Create the dictionary again of Topic names and Topics

Topic_names = {
    "Bank account services" : 0,
    "Credit Card/Prepaid Card" : 1,
    "Mortgages/loans" : 2,
    "Theft/Dispute reporting" : 3,
    "Others" : 4
}
#Replace Topics with Topic Names
df_clean['Topic'] = df_clean['Topic'].map(Topic_names)


# In[66]:


display(df_clean)


# In[67]:


#Keep the columns"complaint_what_happened" & "Topic" only in the new dataframe --> training_data
training_data=df_clean[["complaint_what_happened", "Topic"]]


# In[68]:


display(training_data)


# #### Apply the supervised models on the training data created. In this process, you have to do the following:
# * Create the vector counts using Count Vectoriser
# * Transform the word vecotr to tf-idf
# * Create the train & test data using the train_test_split on the tf-idf & topics
# 

# In[69]:


#Write your code to get the Vector count
count_vector = CountVectorizer(ngram_range=(1,3), stop_words='english', max_df=0.95, min_df=0.02)
vector = count_vector.fit_transform(training_data['complaint_what_happened'])
display(vector.toarray())

#Write your code here to transform the word vector to tf-idf

tfidf_transformer =TfidfTransformer(use_idf=True).fit(vector)
word_vect = tfidf_transformer.transform(vector)
display("\nTransformed Word vector to Tf-idf : {}".format(word_vect.shape))


# You have to try atleast 3 models on the train & test data from these options:
# * Logistic regression
# * Decision Tree
# * Random Forest
# * Naive Bayes (optional)
# 
# **Using the required evaluation metrics judge the tried models and select the ones performing the best**

# In[70]:


# Function to evaluate metrics
from sklearn import metrics

def accuracy_evaluation(y_actual,y_pred,model_name):

    # print classification report of classifier
    display(f"CLASSIFICATION REPORT for {model_name}\n")
    display(metrics.classification_report(y_actual, y_pred, target_names=["Bank Account services", "Credit card or prepaid card", "Others", "Theft/Dispute Reporting",
"Mortgage/Loan"]))

    # Confusion matrix
    plt.title(f"CONFUSION MATRIX for {model_name}\n")
    confusion = metrics.confusion_matrix(y_actual, y_pred)

    sns.heatmap(confusion, annot=True, cbar=None, cmap="Greens", fmt='d', xticklabels=["Bank Account services", "Credit card or prepaid card", "Others", "Theft/Dispute Reporting",
"Mortgage/Loan"], yticklabels=["Bank Account services", "Credit card or prepaid card", "Others", "Theft/Dispute Reporting",
"Mortgage/Loan"])
    plt.show()

    # Metrics calculation
    accuracy = metrics.accuracy_score(y_actual,y_pred)
    clf_report = metrics.classification_report(y_actual,y_pred)   # Sklearn classification report

    display("Accuracy   : " + str(accuracy))



    return accuracy



# ### Train Model and Evaluation

# In[71]:


# Train_Df

train_df = pd.DataFrame(word_vect.toarray(),columns=count_vector.get_feature_names_out(),index=training_data.index)
display(
    train_df.head()
)


# In[72]:


## assign the label to be matched topic
train_df["Topic"] = training_data["Topic"]


# In[73]:


display(train_df.head())


# In[74]:


# Train Test Split

from sklearn.model_selection import train_test_split

X= train_df.drop(['Topic'], axis=1)
y= train_df['Topic']

xtrain, xtest, ytrain, ytest = train_test_split(X, y, train_size=0.70, test_size=0.30, random_state=40, stratify=y)


# ## Building Model and evaluate them using the required metrics

# ### Model 1 : Logistic Regression

# In[75]:


from sklearn.linear_model import LogisticRegression
logistic_regression = LogisticRegression(solver='lbfgs', max_iter=1000)
logistic_regression.fit(xtrain, ytrain)

ytrain_pred = logistic_regression.predict(xtrain)
ytest_pred = logistic_regression.predict(xtest)

display("Training data accuracy\n")
training_accuracy = accuracy_evaluation(ytrain.values,ytrain_pred,'Logistic Regression')


# In[76]:


display("Test data accuracy\n")
test_accuracy = accuracy_evaluation(ytest.values,ytest_pred,'Logistic Regression')


# #### Insights for Model 1:
# Training Accuracy: 96.04% </BR>
# Test Accuracy: 94.7%

# ### Model 2: Naive Bayes

# In[77]:


from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
model_name = 'NAIVE BAYES'
nb = MultinomialNB()
nb.fit(xtrain, ytrain)
y_pred_nb = nb.predict(xtest)


# In[78]:


# Hyperparameter tuning for best result
param_nb = {
    'alpha': (1, 0.1, 0.01, 0.001, 0.0001, 0.00001),
    'fit_prior':[True, False]
}

grid_nb = GridSearchCV(estimator=nb,
                       param_grid=param_nb,
                       verbose=1,
                       scoring='f1_weighted',
                       n_jobs=-1,
                       cv=10)
grid_nb.fit(xtrain, ytrain)
display(grid_nb.best_params_)


# In[79]:


# Creating model with best hyperparameter
model_name = 'NAIVE BAYES'
nb_tuned = MultinomialNB(alpha=1,fit_prior=False)
nb_tuned.fit(xtrain, ytrain)
ypred_train_nb_tuned = nb_tuned.predict(xtrain)
ypred_test_nb_tuned = nb_tuned.predict(xtest)


# In[80]:


# Calculate F1 Score of model using weighted average method
f1_nb = metrics.f1_score(ytrain.values, ypred_train_nb_tuned, average="weighted")
display("Training f1_score : {}".format(f1_nb))

# Calculate F1 Score of model using weighted average method
f1_nb = metrics.f1_score(ytest.values, ypred_test_nb_tuned, average="weighted")
display("Test f1_score : {}".format(f1_nb))


# In[81]:


# Evaluate the Naive Bayes classifier
display("Training Accuracy\n")
Training_acc = accuracy_evaluation(ytrain.values, ypred_train_nb_tuned, "Naive Bayes")


# In[82]:


# Evaluate the Naive Bayes classifier
display("Test Accuracy\n")
Test_acc = accuracy_evaluation(ytest.values, ypred_test_nb_tuned, "Naive Bayes")


# #### Insights for Model 2:
# Training Accuracy: 85.05% </BR>
# Test Accuracy: 85.44%

# ### Model 3 : Decision Tree Classifier

# In[83]:


from sklearn.tree import DecisionTreeClassifier

# Checking initial tree with standard params
dt = DecisionTreeClassifier(random_state=40, max_depth=10, min_samples_leaf = 10, class_weight="balanced")
dt.fit(xtrain,ytrain)

# predict the train and test
ytrain_pred_dt  = dt.predict(xtrain)
ytest_pred_dt = dt.predict(xtest)

# check for model accuracy
display("Train Accuracy")
train_accuracy = accuracy_evaluation(ytrain.values, ytrain_pred_dt,"Decision Tree Classifier")


# In[84]:


# test data accuracy
display("Test Accuracy")
test_accuracy_dt = accuracy_evaluation(ytest.values,ytest_pred_dt,"Decision Tree Classifier")


# #### Insights for Model 3:
# Training Accuracy: 81.99% </BR>
# Test Accuracy: 78.4%

# #### Hyper Parameter Tuning Decision Tree Classifier

# In[85]:


dt = DecisionTreeClassifier(random_state = 40, class_weight="balanced")

# Create the parameter grid based on the results of random search
params = {
    'max_depth': [5, 10, 15, 20],
    'min_samples_leaf': [5, 10, 20, 50],
    'criterion': ["entropy", "gini"]
}

grid_search = GridSearchCV(estimator = dt,param_grid = params,cv = 5, n_jobs=-1, verbose=1, scoring="accuracy", return_train_score=True)

display(grid_search.fit(xtrain, ytrain))


# In[86]:


# Get the results in tabular format
scores = grid_search.cv_results_
display(pd.DataFrame(scores).head())


# In[87]:


display(grid_search.best_params_)


# In[88]:


dt_best = DecisionTreeClassifier(random_state=40, max_depth = 20, min_samples_leaf = 10, class_weight="balanced")
dt_best.fit(xtrain,ytrain)

ytrain_pred_dt = dt_best.predict(xtrain)
ytest_pred_dt = dt_best.predict(xtest)


# In[89]:


display("Training Data evaluation")
train_accuracy_dt = accuracy_evaluation(ytrain.values,ytrain_pred_dt,"DT Classifier Tuned")


# In[90]:


display("test Data evaluation")
test_accuracy_dt = accuracy_evaluation(ytest.values,ytest_pred_dt,"DT Classifier Tuned")


# #### Insights - Decision Tree Classifier Tuned:
# Training Accuracy: 86.49% </BR>
# Test Accuracy: 81.06%

# ### Model 4 : Random Forest Classifier

# In[91]:


from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold


# In[92]:


rf = RandomForestClassifier(random_state=42, class_weight="balanced")

folds = StratifiedKFold(n_splits=4, shuffle = True, random_state=40)

param_grid = {
     'max_depth': [5, 10, 20],
     'min_samples_leaf': [10, 20, 50, 100],
     'n_estimators': [15, 20, 25, 30],
     'min_samples_split': range(10, 30, 50),
     'max_features': [5, 10, 15],
     'criterion': ["gini"]
}

# Instantiate the grid search model
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = folds, verbose = 1, scoring="accuracy", return_train_score=True)


# In[93]:


# Fitting  grid search to the data
display(grid_search.fit(xtrain, ytrain))


# In[94]:


display(grid_search.best_score_)

rf_best = grid_search.best_estimator_
display(rf_best)


# In[95]:


display(rf_best.fit(xtrain,ytrain))


# In[96]:


ytrain_pred_rf = rf_best.predict(xtrain)
ytest_pred_rf = rf_best.predict(xtest)


# In[97]:


display("Training Accuracy")

train_accuracy_rf = accuracy_evaluation(ytrain.values,ytrain_pred_rf,"Random Forest")


# In[98]:


display("Test Accuracy")

test_accuracy_rf = accuracy_evaluation(ytest.values,ytest_pred_rf,"Random Forest")


# #### Insights for Model 4:
# Training Accuracy: 89.78% </BR>
# Test Accuracy: 86.34%

# ### Overall Insights:
# Logistic Regression provides better accuracy and performs better than other models</BR>
# Training Accuracy: 96.04% </BR>
# Test Accuracy: 94.7%

# ### Model Inference using Logistic Regression

# In[99]:


Topic_names_index = {}
for key in Topic_names:
    Topic_names_index[Topic_names[key]] = key
display(Topic_names_index)


# In[100]:


def Topic_Predictor(sentence, model = logistic_regression):
    vect_custom = count_vector.transform(pd.Series(sentence))
    word_vect_custom = tfidf_transformer.transform(vect_custom)

    word_vect_custom_df = pd.DataFrame(word_vect_custom.toarray(),columns=count_vector.get_feature_names_out()) #Convert to data frame
    custom_pred = model.predict(word_vect_custom_df) # Make predictions

    return Topic_names_index[custom_pred[0]]


# In[101]:


custom_text = pd.DataFrame({'complaints': ["I can not get from chase who services my mortgage, who owns it and who has original loan docs",
                                  "The bill amount of my credit card was debited twice. Please look into the matter and resolve at the earliest.",
                                  "I want to open a salary account at your downtown branch. Please provide me the procedure.",
                                  "unwanted service activated and money deducted automatically ",
                                  "How can I know my CIBIL score?",
                                  "Where are the bank branches in the city of Patna?",
                                 "This letter is to dispute an incorrectly charged amount on my credit card. My credit card number is XCXCXXC and it has a constant charges"
                                          ,"I want to report an incident of theft of my credit card.",
                                          "I would like to convert my account into premium account to avail latest benefits.",
                                          "I am buying a new house, based on my current salary I should get a good loan."]})
display(custom_text)


# In[102]:


custom_text['predicted topic'] = custom_text['complaints'].apply(lambda x: Topic_Predictor([x]))
display(
    custom_text
)


# ## Conclusion : </br>
# Logistic Regression provides better accuracy and performs better than other models

# In[103]:


import datetime, pytz;
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

