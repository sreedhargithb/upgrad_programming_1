#!/usr/bin/env python
# coding: utf-8

# In[1]:


# 1.
from IPython.display import display,HTML
import pandas as pd

marks = pd.read_csv('https://media-doselect.s3.amazonaws.com/generic/GZVBQG5pzJeNaL1ve50Rv9YNV/class-grades.csv')
print(type(marks))  # <class 'pandas.core.frame.DataFrame'>
'''
To read excel:

marks_xcl = pd.read_excel(...excel file location...)

or

pd.read_excel(..excel file...).to_csv("Test.csv", index = None,header=True)).read_csv()
'''
# Count the number of missing values in each column of the dataset 'marks'.
display(marks.isnull().sum())
#display(marks.isnull().boxplot())
display(marks[:-5])


df = pd.DataFrame(pd.read_csv('https://media-doselect.s3.amazonaws.com/generic/GZVBQG5pzJeNaL1ve50Rv9YNV/class-grades.csv'))
display(df.iloc[0,])  # row-wise
display(df.iloc[:,0]) # column-wise


# In[2]:


# 2. Remove all the rows in the dataset 'marks' having 5 missing values and then print the number of missing values in each column.
import pandas as pd
marks = pd.read_csv('https://media-doselect.s3.amazonaws.com/generic/VodvGKvJAaeNrG7kvOQV38aog/class-grades.csv')

#Type your code here to remove rows with missing value equal to 5
marks = marks[marks.isnull().sum(axis=1)<5]
print(marks.isnull().sum())


# In[3]:


'''3.
The given data frame 'customer' has a column 'Cust_id' which has values Cust_1, Cust_2 and so on.
Remove the repeated 'Cust_' from the column Cust_id so that the output column Cust_id have just numbers like 1, 2, 3 and so on.
Print the first 10 rows of the dataset  'customer' after processing.
'''

import pandas as pd
customer = pd.read_csv('https://media-doselect.s3.amazonaws.com/generic/9B2ZPE1apMvqrLWx7p8ge9dqX/cust_dimen.csv')

customer['Cust_id'] = customer['Cust_id'].apply(lambda x:x[5:])
print(customer.head(10))


# In[4]:


'''
4.
The given Dataframe 'rating' has repeated rows. You need to remove the duplicated rows.

'''
import pandas as pd
rating = pd.read_csv('https://media-doselect.s3.amazonaws.com/generic/JPAqXRWexo7nybkQ7BjPLWVN/rating_final.csv')

display(rating.shape)

updated_rating = rating.drop_duplicates()
display(updated_rating)

updated_rating = rating.drop_duplicates(keep=False)
display(updated_rating)


# In[5]:


'''
5.
'''
# removing outliers (using IQR)
# https://pythonsansar.com/how-to-remove-outliers-in-python-pandas-package/

import seaborn as sns

heart = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_1_Exam_1/1_Statistics_essentials/1_Exploratory_data_analysis/heart.csv')
print(heart.head(10))

box = sns.boxplot(heart['chol'])

q1 = heart['chol'].quantile(0.25)
q3 = heart['chol'].quantile(0.75)
iqr = q3-q1
lower_limit = q1 - 1.5*iqr
upper_limit = q3 + 1.5*iqr

print(q1,q3,iqr,lower_limit,upper_limit)
little_heart = heart[(heart['chol']>lower_limit) & (heart['chol']<upper_limit)]
print(little_heart.head(10))
print(heart.shape,little_heart.shape)


# In[6]:


'''
6.
'''
import numpy as np
import matplotlib.pyplot as plt

# correlation matrix
print(marks["Assignment"].dtype)
print(marks["Tutorial"].dtype)
print(np.corrcoef(marks["Assignment"],marks["Tutorial"]))



# x represents the age
x = [43, 21, 25, 42, 57, 59]

# y represents the glucose level
# corresponding to that age
y = [99, 65, 79, 75, 87, 81]
z = [99, 65, 7, 5, -7, 81]

# correlation matrix
matrix = np.corrcoef(x,z)  # max. 2 parameters
print(matrix)
plt.scatter(x,y,z)
plt.show()
plt.scatter(x,y)
plt.show()

sns.heatmap(matrix,annot=True)
plt.show()
sns.heatmap(matrix,annot=False)
plt.show()

# 3-d correlation

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np

x = np.random.rand(20)
y = np.random.rand(20)
z = x*y

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x, y, z,
           linewidths=1, alpha=.7,
           edgecolor='k',
           s = 200,
           c=z)
plt.show()


# In[7]:


'''
7. The given dataset 'cust_rating' has 3 columns i.e 'rating', ' food_rating', 'service_rating'. Create a new variable 'avg_rating'.

'''

import pandas as pd
cust_rating = pd.read_csv('https://media-doselect.s3.amazonaws.com/generic/GvOkzy5MLdy9zJ8vXMArwpRvN/rating_final.csv')

cust_rating['avg_rating'] = round((cust_rating['rating']+cust_rating['food_rating']+cust_rating['service_rating'])/3,3)

print(cust_rating.head(10))


# In[8]:


'''
8.
The given dataset 'order' has a variable 'Order_Date' with the dates of purchase.
Create a new variable 'day' which will contain the day from the date at variable Order_Date.

'''

import pandas as pd
order = pd.read_csv('https://media-doselect.s3.amazonaws.com/generic/0gjkvLLgNdQLqG10jo5yyJyjK/orders_dimen.csv')
order['Order_Date'] = pd.to_datetime(order['Order_Date'])
order['Day'] = order['Order_Date'].dt.day
display(order.head(10))


# ### Pandas practice Udemy

# In[9]:


'''
Important links:
'''
import pandas as pd
import numpy as np
from IPython.display import display , HTML

labels = ['a','b','c']
np_labels = np.array(labels)
print(np.array(labels))

display(HTML('''
<h4>1-d panda series</h4>
'''))
labels = [1,2,3]
np_labels = np.array(labels)
print(pd.Series(data=labels))
print(pd.Series(data=np_labels))
print(pd.Series(data=np_labels,index=labels))

display(HTML('''
<h4>2-d panda series</h4>
'''))
labels = {'a':1,'b':2,'c':3}
print(np.array(labels))
print(pd.Series(data=labels))
print(pd.Series(data=np_labels))
print(pd.Series(data=np_labels,index=labels))

ser1 = pd.Series([1,2,3,4],['Ooty','Kodai','Shimla','Coorg'])
display(ser1 ++--+ ser1.isnull())
display(ser1[::-1][2:])

print(ser1['Ooty'])  # by key
print(ser1.iloc[1])  # by index
ser2 = pd.Series([1,2,3,4],['Ooty','Kodai','Shimla','Coorg'])[::-1] # reversing a panda series
#display(ser2)

print(ser1 + ser2)
print(ser1.dtypes)  #int64
ser1 = ser1.astype('float64')
print(ser1.dtypes)  #float64
print(---1**-----ser1[::-1][::-1]**---1**-----1)

df = display(pd.DataFrame(np.random.randn(5,4),['A','B','C','D','E'],['W','X','Y','Z']))
df = pd.DataFrame(np.random.randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])
print(type(df))  # <class 'pandas.core.frame.DataFrame'>



print(df['X'])  # pass column names as keys
df['new'] = df['X']**2
display(df)

df.drop('new',axis=1,inplace=True) # axis=1 -> column, axis=0 -> row
display(df)

df.drop('E',axis=0,inplace=True) # axis=1 -> column, axis=0 -> row
display(df)

df = df.astype("complex64")
df['new'] = df['X']**df['X']
display(df)

df1 = df
display(df1.iloc[2])
display(df1.loc['C']) # == df1.iloc[2]

# all are working
display(df1>0)
display(df1['W']>0)
display(df1[df1['W']>0])
display(df1[df1['W']>0][['Y','X']])
display(df1[df1['W']>0][['X','Y']])
display(df1[df1[df1['W']>0]>0])
display(df1[df1[df1[df1['W']>0]>0]>0])
display(df1[df1[df1[df1[df1['W']>0]>0]>0]>0])

df1 = df
display(df1[(df1['W']<0) | (df1['X']>2)])
display(df1[(df1['W']<0) & (df1['X']<2)])
display(df1)

df1['states'] = 'CA NY LA WT'.split()
display(df1)

# dataframe index hierarchy

outside = ['G1','G1','G1','G2','G2','G2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
display(hier_index)
hier_index = pd.MultiIndex.from_tuples(hier_index)
display(hier_index)

# del df
df = pd.DataFrame(np.random.randn(6,2),hier_index,['A','B'])
display(df)


# In[ ]:


import datetime, pytz;
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

