import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import datetime as dt

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from scipy.cluster.hierarchy import cut_tree

# read the dataset
retail_df = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_2_Exam_2/3_Machine_learning_2/7_Unsupervised_learning__clustering/1_K-means/_1_Lab/Online_Retail.csv", sep=",", encoding="ISO-8859-1", header=0)
print(retail_df.head())

# basics of the df
print(retail_df.info())

# missing values
print(round(100*(retail_df.isnull().sum())/len(retail_df), 2))

# drop all rows having missing values
retail_df = retail_df.dropna()
print(retail_df.shape)

print(retail_df.head())

# new column: amount 
retail_df['amount'] = retail_df['Quantity']*retail_df['UnitPrice']
print(retail_df.head())

# monetary
grouped_df = retail_df.groupby('CustomerID')['amount'].sum()
grouped_df = grouped_df.reset_index()
print(grouped_df.head())

# frequency
frequency = retail_df.groupby('CustomerID')['InvoiceNo'].count()
frequency = frequency.reset_index()
frequency.columns = ['CustomerID', 'frequency']
print(frequency.head())

# merge the two dfs
grouped_df = pd.merge(grouped_df, frequency, on='CustomerID', how='inner')
print(grouped_df.head())

print(retail_df.head())

# recency
# convert to datetime
retail_df['InvoiceDate'] = pd.to_datetime(retail_df['InvoiceDate'], 
                                          format='%d-%m-%Y %H:%M')

print(retail_df.head())

# compute the max date
max_date = max(retail_df['InvoiceDate'])
print(max_date)

# compute the diff
retail_df['diff'] = max_date - retail_df['InvoiceDate']
print(retail_df.head())

# recency
last_purchase = retail_df.groupby('CustomerID')['diff'].min()
last_purchase = last_purchase.reset_index()
print(last_purchase.head())

# merge
grouped_df = pd.merge(grouped_df, last_purchase, on='CustomerID', how='inner')
grouped_df.columns = ['CustomerID', 'amount', 'frequency', 'recency']
print(grouped_df.head())

# number of days only
grouped_df['recency'] = grouped_df['recency'].dt.days
print(grouped_df.head())

# 1. outlier treatment
plt.boxplot(grouped_df['recency'])

# two types of outliers:
# - statistical
# - domain specific

# removing (statistical) outliers
Q1 = grouped_df.amount.quantile(0.05)
Q3 = grouped_df.amount.quantile(0.95)
IQR = Q3 - Q1
grouped_df = grouped_df[(grouped_df.amount >= Q1 - 1.5*IQR) & (grouped_df.amount <= Q3 + 1.5*IQR)]

# outlier treatment for recency
Q1 = grouped_df.recency.quantile(0.05)
Q3 = grouped_df.recency.quantile(0.95)
IQR = Q3 - Q1
grouped_df = grouped_df[(grouped_df.recency >= Q1 - 1.5*IQR) & (grouped_df.recency <= Q3 + 1.5*IQR)]

# outlier treatment for frequency
Q1 = grouped_df.frequency.quantile(0.05)
Q3 = grouped_df.frequency.quantile(0.95)
IQR = Q3 - Q1
grouped_df = grouped_df[(grouped_df.frequency >= Q1 - 1.5*IQR) & (grouped_df.frequency <= Q3 + 1.5*IQR)]



# 2. rescaling
rfm_df = grouped_df[['amount', 'frequency', 'recency']]

# instantiate
scaler = StandardScaler()

# fit_transform
rfm_df_scaled = scaler.fit_transform(rfm_df)
print(rfm_df_scaled.shape)

rfm_df_scaled = pd.DataFrame(rfm_df_scaled)
rfm_df_scaled.columns = ['amount', 'frequency', 'recency']
print(rfm_df_scaled.head())

# k-means with some arbitrary k
kmeans = KMeans(n_clusters=4, max_iter=50)
kmeans.fit(rfm_df_scaled)

print(kmeans.labels_)

# help(KMeans)

# elbow-curve/SSD
ssd = []
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]
for num_clusters in range_n_clusters:
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    ssd.append(kmeans.inertia_)
    
# plot the SSDs for each n_clusters
# ssd
plt.plot(ssd)

# silhouette analysis
range_n_clusters = [2, 3, 4, 5, 6, 7, 8]

for num_clusters in range_n_clusters:
    
    # intialise kmeans
    kmeans = KMeans(n_clusters=num_clusters, max_iter=50)
    kmeans.fit(rfm_df_scaled)
    
    cluster_labels = kmeans.labels_
    
    # silhouette score
    silhouette_avg = silhouette_score(rfm_df_scaled, cluster_labels)
    print("For n_clusters={0}, the silhouette score is {1}".format(num_clusters, silhouette_avg))
    
    

# final model with k=3
kmeans = KMeans(n_clusters=3, max_iter=50)
kmeans.fit(rfm_df_scaled)

print(kmeans.labels_)

# assign the label
grouped_df['cluster_id'] = kmeans.labels_
print(grouped_df.head())

# plot
sns.boxplot(x='cluster_id', y='amount', data=grouped_df)

print(rfm_df_scaled.head())

print(grouped_df.head())

# single linkage
mergings = linkage(rfm_df_scaled, method="single", metric='euclidean')
dendrogram(mergings)
plt.show()

# complete linkage
mergings = linkage(rfm_df_scaled, method="complete", metric='euclidean')
dendrogram(mergings)
plt.show()

# 3 clusters
cluster_labels = cut_tree(mergings, n_clusters=3).reshape(-1, )
print(cluster_labels)

# assign cluster labels
grouped_df['cluster_labels'] = cluster_labels
print(grouped_df.head())

# plots
sns.boxplot(x='cluster_labels', y='recency', data=grouped_df)

# plots
sns.boxplot(x='cluster_labels', y='frequency', data=grouped_df)

# plots
sns.boxplot(x='cluster_labels', y='amount', data=grouped_df)

import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))
