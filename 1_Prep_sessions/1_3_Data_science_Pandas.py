#!/usr/bin/env python
# coding: utf-8

# ## Contents:
# 
# 
# 1. [Intro_to_dataframes](#Intro_to_dataframes)<br>
# 2. [Operations_on_Pandas](#Operations_on_Pandas)<br>
# 3. [Practice_exercise_1](#Practice_exercise_1)<br>
# 4. [Seoul_dataset](#Seoul_dataset)<br>
# 5. [Sales_dataset](#Sales_dataset)<br>
# 5. [Pandas_Udemy](#Pandas_Udemy)<br>
#     

# <h2 style = "color : Brown"> Data Frame </h2>
# <a id='Intro_to_dataframes'>Intro_to_dataframes</a>

# In[9]:


# All imports
import numpy as np
import pandas as pd
import os
print(os.getcwd()) # Shows where you are
print(os.listdir()) # Shows what files are actually there


# <h4 style = "color : Sky blue"> Example - 1</h4>  
# 
# ##### Create a Data Frame cars using raw data stored in a dictionary
# 

# In[10]:


cars_per_cap = [809, 731, 588, 18, 200, 70, 45]
country = ['United States', 'Australia', 'Japan', 'India', 'Russia', 'Morocco', 'Egypt']
drives_right = [True, False, False, False, True, True, True]


# In[11]:


data = {"cars_per_cap": cars_per_cap, "country": country, "drives_right": drives_right}


# In[12]:


data


# In[13]:


cars = pd.DataFrame(data)

cars


# In[14]:


type(cars)


# <h4 style = "color : Sky blue"> Example - 2 (Reading data from a file)</h4>  
# 
# ##### Create a Data Frame by importing cars data from cars.csv

# In[15]:


# Read a file using pandas

cars_df = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/Movies.csv')
cars_df


# <h4 style = "color : Sky blue"> Example - 3 (Column headers)</h4>  
# 
# ##### Read file - skip header

# In[16]:


cars_df = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/cars.csv', header=None)

cars_df


# ##### Assign Headers

# In[17]:


# Returns an array of headers

cars_df.columns


# In[18]:


# Rename Headers

cars_df.columns = ['country code', 'region', 'country', 'cars_per_cap', 'drive_right']


# In[19]:


cars_df


# <h4 style = "color : Sky blue"> Example - 4 (Row index/names) </h4>  
# 
# ##### Read file - skip header and assign first column as index.

# In[20]:


# Index is returned by
cars_df.index


# In[21]:


# Read file and set 1st column as index
cars_df = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/cars.csv", header= None, index_col=0)

# set the column names
cars_df.columns = ['region', 'country', 'cars_per_cap', 'drive_right']
cars_df


# In[22]:


# Print the new index
cars_df.index


# ##### Rename the Index Name

# In[23]:


cars_df.index.name = 'country_code'
cars_df


# ##### Delete the index name

# In[24]:


cars_df.index.name = None
cars_df


# ##### Set Hierarchical index

# In[25]:


# Read file and set 1st column as index
cars_df = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/cars.csv", header= None)

# set the column names
cars_df.columns = ['country_code','region','country','cars_per_cap','drives_right']

cars_df.set_index(['region', 'country_code'], inplace=True)


# In[26]:


cars_df


# In[ ]:





# <h4 style = "color : Sky blue"> Example - 5 (Write Data Frame to file) </h4>  
# 
# ##### Write cars_df to cars_to_csv.csv

# In[27]:


cars_df.to_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/cars_to_csv.csv')


# In[28]:


#Marks.csv


import numpy as np
import pandas as pd

# The file is stored at: 'https://media-doselect.s3.amazonaws.com/generic/A08MajL8qN4rq72EpVJbAP1Rw/marks_1.csv'
# Provide your answer below
df = pd.read_csv('https://media-doselect.s3.amazonaws.com/generic/A08MajL8qN4rq72EpVJbAP1Rw/marks_1.csv', header = None, sep = '|')

print(df)


# In[29]:


df = pd.read_csv('https://media-doselect.s3.amazonaws.com/generic/A08MajL8qN4rq72EpVJbAP1Rw/marks_1.csv', sep='|',header=None) # Write your answer here
df.columns = ['S.No','Name','Subject','Maximum Marks','Marks Obtained','Percentage']
df=df.set_index('S.No')
df


# In[30]:


# Conditional statement in dataframes

import pandas as pd
df = pd.read_csv('https://cdn.upgrad.com/uploads/production/b3467ba4-4e13-44e9-8087-4d7e94cc7586/forestfires.csv')
df_2 = df.loc[(df.area>0)&(df.temp>15)&(df.wind>1),:]
print(df_2.head(20))


# In[31]:


### Create DataFrames 

#Since a new concept is being introduced, it is beneficial to explore the concept first using simple DataFrames. Once you understand the usage and the capabilities of these concepts, you can think of ways to apply these capabilities as and when needed. 


import pandas as pd

df_1 = {"col1":[1,2,3,4], "col2": [5,6,7,8]}
df_2 = {"col1":[11,12,13,14], "col2": [15,16,17,18]}

df1 = pd.DataFrame(df_1)
df2 = pd.DataFrame(df_2)

df1

df2

### Concatenation 

# It is used when you want to stick two dataframes together without any consideration given to matching elements. In contrast, the merge command uses a key to stitch two data frames together. 

# If the shape of the two concatenating dataframes does not match, NaN values are added to make the dimensions uniform. 


pd.concat([df1, df2], axis = 0)

# Axis 0 represents row wise concatenation

# **NOTE**

# - Rows in df2 get added to the df1
# - Intexes of df2 remain the same as they were before the join. 

pd.concat([df1, df2], axis = 1)

# Axis 0 represents column wise concatenation

df1["col3"] = df1["col1"] + df1["col2"]

# After this operation df1 will have 3 columns while df2 has only 2. 

pd.concat([df1, df2], axis = 0)

# Since there is one extra column in df1, the corresponding vales in df2 become `NaN` or null values. 

# ### Arithmetic Operators on DataFrames

# You can perform element wise operations on dataframes as well. These are very similar to operations you performed on NumPy arrays. 

# for example, if you want to add all the elements on `df1` to the correspopnding elements on `df2` you can use the '+' operator. 

df1 + df2 

# As you saw all the elements in `df1` got added to corresponding elements in `df2`

# But the `df1` had three columns while `df2` had two. So the operation for the third column is incomplete, that is why you see the null values in the result. This is the most significant difference in using operators in pandas and NumPy; this operation would have thrown an error if it was executed using NumPy arrays.  

# The same result can be achieved by the `add()` method

# df1.add(df2)

# Along with the normal addition this add method also provides additional functionalities. You can read about them [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.add.html)

# similar to the '+' operator and the `add()` there are other operators as well 

# - `sub()`: ' - '
# - `mul()`: ' * '
# - `div()`: ' / '
# - `floordiv()`: ' // '
# - `mod()`: ' % ' 
# - `pow()`: ' ** '

# recreating the DataFrames so that the dimentions match. 

df_1 = {"col1":[1,2,3,4], "col2": [5,6,7,8]}
df_2 = {"col1":[11,12,13,14], "col2": [15,16,17,18]}

df1 = pd.DataFrame(df_1)
df2 = pd.DataFrame(df_2)

print (df1)
print (df2)

df2 - df1

df2 ** df1

# recreating the DataFrames so that the dimentions match. 

df_1 = {"col1":[1,2,3,4], "col2": [5,6,7,8]}
df_2 = {"col1":[11,12,13,14]}

df1 = pd.DataFrame(df_1)
df2 = pd.DataFrame(df_2)

print (df1)
print (df2)

df1 + df2

# One of the advantages of pandas DataFrame is that it can hold data of different data types. 

# Which leads us to the question What would happen of operators were used on DataFrames which have "non-numerical" data types?

df_1 = {"col1":[1,2,3,4], "col2": [5,6,7,8], "col3": [True,False,False,True], "col4": ["a","b","c","d"] }
df_2 = {"col1":[11,12,13,14], "col2": [15,16,17,18], "col3": [True,False,True,False], "col4": ["e","f","g","h"]}

df1 = pd.DataFrame(df_1)
df2 = pd.DataFrame(df_2)

print (df1)
print (df2)

df1 +df2 

# Something very interesting has happened. 

# Pandas was smart enough to recognise the different data types and use the operators accordingly. 

# - For int data type, it performed addition 
# - For boolean, it performed OR operation
# - For string, it performed concatenation 

# The below expression throws an error because there is not '-' in strings and pandas cannot figure out what to do. 
# df1 - df2


df_1 = {"col1":[1,2,3,4], "col2": [5,6,7,8], "col3": [True,False,False,True], "col4": ["a","b","c","d"] }
df_2 = {"col1": [True,False,True,False], "col2": ["e","f","g","h"], "col3":[11,12,13,14], "col4": [15,16,17,18] }

df1 = pd.DataFrame(df_1)
df2 = pd.DataFrame(df_2)

print (df1)
print (df2)

# Since the data types of correcponding columns do not match Pandas throws a type error for the below expression
# df1 + df2


# ### Summary

# ##### 1. `Concatenation` : Used when you want to stich to dataframes together without any reguard to the values. 
# a. Even if the shapes do not match the operation is performed. Filling Null values wherever necessary. 
# ##### 2. `operators` : Can perform element wise operations on Pandas DataFrames. 
# a. You can use operators themselves '+' or the function `add()` for the same result.  
# b. If the Shape does not match then null values are added. 
# c. Can work with differnet data types as well, as long as the operation is defined for that data type. 


# <h2 style = "color : Brown"> Operations on Pandas</h2>
# <a id='Operations_on_Pandas'>Operations_on_Pandas</a>

# <a id='Practice_exercise_1'>Practice_exercise_1</a>

# In[32]:


# Import necessary libraries
import numpy as np
import pandas as pd
from IPython.display import display

# Task 1: Reading and Inspection

# Subtask 1.1: Import and read the dataset
movies = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/Movies.csv")

# Display the first few rows of the dataframe
display(movies)

# Check the number of missing values across the columns
print((movies.isnull().sum() > 0).sum())  # Count of columns with missing values
print(movies.shape)  # Shape of the dataframe
print(movies.isnull().sum())  # Missing values per column
print(movies["language"].isnull().sum())  # Missing values in language column
print(movies["genres"].isnull().sum())  # Missing values in genres column
print(movies["num_critic_for_reviews"].isnull().sum())  # Missing values in critic reviews column
print(movies["imdb_score"].isnull().sum())  # Missing values in imdb_score column

# Count movies with English language
print(len(movies[movies["language"] == "English"]))

# Fill missing language values with 'English'
movies["language"].fillna("English", inplace=True)

# Count movies with English language after filling missing values
print(len(movies[movies["language"] == "English"]))

# Subtask 1.2: Inspect the dataframe
# Show the data types of each column
print(movies.dtypes)

# Show the column names
print(movies.columns)

# Question 1: How many rows and columns are present in the dataframe?
# Answer: The shape of the dataframe is (3879, 28)
# Based on the shape information from above, we know the answer is (3879, 28).

# Question 2: How many columns have null values present in them?
# The count of columns with missing values can be calculated using the following code:
print((movies.isnull().sum() > 0).sum())  # This will return the number of columns with null values.

# Task 2: Cleaning the Data

# Subtask 2.1: Drop unnecessary columns
columns_to_drop = [
    "color", "director_facebook_likes", "actor_1_facebook_likes", 
    "actor_2_facebook_likes", "actor_3_facebook_likes", "actor_2_name", 
    "cast_total_facebook_likes", "actor_3_name", "duration", "facenumber_in_poster", 
    "content_rating", "country", "movie_imdb_link", "aspect_ratio", "plot_keywords"
]

# Drop the columns
movies.drop(columns=columns_to_drop, axis=1, inplace=True)

# Display the new shape of the dataframe
print(movies.shape)  # Question 3: The count of columns in the new dataframe will be 15.

# Subtask 2.2: Inspect Null values
# Calculate the percentage of missing values in each column
null_percentages = movies.isnull().mean() * 100
display(null_percentages)

# Question 4: Which column has the highest percentage of null values?
# You can find the column with the highest percentage of null values by sorting the null percentages:
print(null_percentages.idxmax())  # This will give the column name with the highest null percentage

# Subtask 2.3: Fill NaN values
# As discussed, we will fill the missing values in the 'language' column with "English".
movies["language"].fillna("English", inplace=True)

# Question 5: What is the count of movies made in English after replacing the NaN values with English?
print(len(movies[movies["language"] == "English"]))  # Answer: 3674

# Task 3: Data Analysis

# Subtask 3.1: Change the unit of columns
# Convert the 'budget' and 'gross' columns from dollars to millions of dollars.
movies['budget'] = movies['budget'] / 1e6
movies['gross'] = movies['gross'] / 1e6

# Display the updated columns
display(movies[['budget', 'gross']].head())

# Subtask 3.2: Find the movies with the highest profit

# Create a new column called 'profit' as the difference between 'gross' and 'budget'
movies['profit'] = movies['gross'] - movies['budget']

# Sort the dataframe by 'profit' in descending order
movies_sorted_by_profit = movies.sort_values(by='profit', ascending=False)

# Extract the top 10 profiting movies
top10 = movies_sorted_by_profit.head(10)

# Display the top 10 profiting movies
display(top10)

# Question 6: Which movie is ranked 5th from the top in the list obtained?
# Based on the sorted dataframe, this should be:
print(top10.iloc[4]['movie_title'])  # Get the title of the 5th movie

# Subtask 3.3: Find IMDb Top 250

# Filter movies that have a num_voted_users greater than 25,000 and sort by imdb_score
IMDb_Top_250 = movies[movies['num_voted_users'] > 25000].sort_values(by='imdb_score', ascending=False).head(250)

# Add a 'Rank' column to IMDb_Top_250
IMDb_Top_250['Rank'] = range(1, 251)

# Display the IMDb Top 250 dataframe
display(IMDb_Top_250)

# Question 7: Which bucket holds the maximum number of movies from IMDb_Top_250?
# Create buckets based on the 'imdb_score' column
buckets = pd.cut(IMDb_Top_250['imdb_score'], bins=[7.5, 8, 8.5, 9, 9.5, 10])

# Count the number of movies in each bucket
bucket_counts = buckets.value_counts()

# Display the bucket with the maximum count
print(bucket_counts.idxmax())  # This will print the range with the most movies

# Subtask 3.4: Find the critic-favorite and audience-favorite actors

# Create dataframes for Meryl Streep, Leonardo DiCaprio, and Brad Pitt
Meryl_Streep = movies[movies['actor_1_name'] == 'Meryl Streep']
Leo_Caprio = movies[movies['actor_1_name'] == 'Leonardo DiCaprio']
Brad_Pitt = movies[movies['actor_1_name'] == 'Brad Pitt']

# Combine the three dataframes into one
Combined = pd.concat([Meryl_Streep, Leo_Caprio, Brad_Pitt])

# Group by actor and calculate the mean of critic and user reviews
combined_grouped = Combined.groupby('actor_1_name').agg(
    mean_critic_reviews=('num_critic_for_reviews', 'mean'),
    mean_user_reviews=('num_user_for_reviews', 'mean')
)

# Display the result
display(combined_grouped)

# Question 8: Which actor is highest rated among the three actors according to the user reviews?
# Answer: The actor with the highest mean user reviews
highest_rated_user = combined_grouped['mean_user_reviews'].idxmax()
print(highest_rated_user)

# Question 9: Which actor is highest rated among the three actors according to the critics?
# Answer: The actor with the highest mean critic reviews
highest_rated_critic = combined_grouped['mean_critic_reviews'].idxmax()
print(highest_rated_critic)



# <a id='Seoul_dataset'>Seoul_dataset</a>

# In[33]:


# ####  Seoul dataset
# #Question

# # The air quality data for this segment has been divided into three different csv files. 

# # `info.csv` has the data hour by hour data about the concentration of polutants in the air and the status of the intruments. 
# # `item_info` has the data for items and levels of concetration. 
# # `station_info` has the data for measutring stations. 

# # Read in all the three datasets and then print the first five rows.

# # You can download the dataset from kaggle website: https://www.kaggle.com/bappekim/air-pollution-in-seoul

# import pandas as pd

# data = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/Measurement_info.csv", header = 0)
# data

# item = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/Measurement_item_info.csv", header = 0)
# item

# station = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/Measurement_station_info.csv", header = 0)
# station

# #### Question

# # Create a new Dataframe whcih has information about the item code and item name

# sub_item = item[['Item code', 'Item name']]
# sub_item

# #### Question

# # Create a new Dataframe whcih has information about the station code and station name

# sub_station = station[['Station code', 'Station name(district)']]
# sub_station

# ### Question 

# # In the `data` DataFrame add in a column displaying the names of the items. 

# data_i = data.merge(sub_item, on = "Item code", how = "left")
# data_i.head()

# ### Question 

# # In the `data_i` DataFrame add in a column displaying the names of the stations. 

# data_s = data_i.merge(sub_station, on = "Station code", how = "left")
# data_s.head()

# ### Question 

# # In the `data_s` DataFrame drop the columns `Station code` and `Item code`. As these columns have not become redundant. You can find the relevant function in the pandas library [here](https://pandas.pydata.org/docs/reference/index.html). 

# data = data_s.drop(['Station code', 'Item code'], axis = 1)
# data.head()

# ### Question 

# # Given below are the meanings of the values in the `Instrument status`. 

# # - 0: Normal 
# # - 1: Need for calibration 
# # - 2: Abnormal
# # - 4: Power cut off 
# # - 8: Under repair
# # - 9: Abnormal data

# # Using the information given above, add a column in the `data` DataFrame to give the status of the intsruments. Then drop the `Instrument status` column.  

# # Hint: First create a dictionary from the data, then use the same dictionary to create a DataFrame and then merge the DataFrame. with `data`

# status_dict = {"Instrument status": [0,1,2,4,8,9], 
#                "Status": ["Normal", "Need for calibration", "Abnormal", "Power cut off", "Under repair", "Abnormal data"]}
# status_dict

# dictdf = pd.DataFrame(status_dict)
# dictdf

# data = data.merge(dictdf, on = "Instrument status", how = "left")

# data.head()

# data = data.drop(["Instrument status"], axis = 1)

# data.head()

# ### Question 

# # Extract the time series data, that is year, month, date and hour form the `Measurement date` column. Once all the data is extrcted drop the `Measurement date` column. 

# # This operation might take some time as the dataset we are working with is very large. 

# data['Year'] = pd.DatetimeIndex(data['Measurement date']).year
# data['Month'] = pd.DatetimeIndex(data['Measurement date']).month
# data['Date'] = pd.DatetimeIndex(data['Measurement date']).day
# data['Hour'] = pd.DatetimeIndex(data['Measurement date']).hour
# data.head()

# data = data.drop(["Measurement date"], axis =1)

# data.head()


# In[34]:


get_ipython().system('pip install openpyxl')


# <a id='Sales_dataset'>Sales_dataset</a>

# In[35]:


from IPython.display import display,HTML

display(HTML('''
<h2 style = "color : Brown">Case Study - Sales Data </h2>
'''))

# All imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

##### Sales and Profit data is read in dataframe "sales"

# Read file 

sales = pd.read_excel('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/sales.xlsx')
sales

# Read file and set 1st two columns as index
sales = pd.read_excel('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/sales.xlsx', index_col = [0,1])

sales

display(HTML('''
<h4 style = "color : Sky blue"> Example - 1</h4>  
'''))

##### Display first 3 land last 3 rows of the sales dataframe


sales.head() # Default - returns top 5 rows

sales.head(3)

sales.tail()

sales.tail(3)

display(HTML('''
<h4 style = "color : Sky blue"> Example - 2</h4>   
'''))

##### Display the information about the data stored in data frame


sales.info()

##### Display the statistical information about the data in dataframe

sales.describe()

sales[["Sales", "Profit"]].plot(kind= "box", subplots= True)
plt.show()

sales["Profit"]


##########################################################################################################3
display(HTML('''
<h2 style = "color : Brown">Case Study - Sales Data </h2>
'''))

# All imports
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Displays pandas float values in 2 decimals
pd.options.display.float_format = '{:,.2f}'.format

sales = pd.read_excel('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/sales.xlsx')
sales

##### Sales and Profit data is read in dataframe "sales"

# Read file and set 2nd column as index

sales = pd.read_excel('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/1_Prep_sessions/pandas_datasets/sales.xlsx', index_col = [1])
sales

display(HTML('''
<h4 style = "color : Sky blue"> Example - 1 (Column Indexing)</h4>  
'''))

##### Display Sales Column

sales["Sales"]

sales.Sales

type(sales["Sales"])

##### Display Sales and Profit Column together

sales[["Sales", "Profit"]]

display(HTML('''
<h4 style = "color : Sky blue"> Example - 2 (Row Indexing)</h4>  
'''))

##### Display data for "Southern Asia"

# loc accessor takes row index and column index

sales.loc["Southern Asia"]

##### Display Sales data for "Southern Asia"

sales.loc["Southern Asia", "Sales"]

##### Display data for "Southern Asia"

# iloc accessor takes row number and column number

sales.iloc[6]

sales.iloc[6,3]

display(HTML('''
<h4 style = "color : Sky blue"> Example - 3 (Slicing)</h4>
'''))  

##### Display data for  Market, Sales and Profit

sales.loc[:, ["Market", "Sales", "Profit"]].head()

sales.iloc[:, [0,3,2] ].head()

##### Display data for Western Africa Southern Africa and North Africa

sales.loc[["Western Africa", "Southern Africa", "North Africa"] ,:]

sales.iloc[0:3, :]

##### Display Sales and Profit data for Western Africa Southern Africa and North Africa

sales.loc[["Western Africa", "Southern Africa", "North Africa"] , ["Sales", "Profit"]]

sales.iloc[0:3, 2:4]

display(HTML('''
<h4 style = "color : Sky blue"> Example - 4 (Filtering)</h4> 
''')) 

##### Display Markets with Sales >300000

sales["Sales"] > 300000

sales[ sales["Sales"] > 300000 ]

##### Display the LATAM and Eruopean countries with sales > 250000

sales[  (sales["Market"].isin(["LATAM", "Europe"])) & (sales["Sales"] > 250000)     ]

### Optional Examples 

# The examples given below are good to know but not essential to achieve the objective of this session. You can go through them at your own pace. 

display(HTML('''
<h4 style = "color : Sky blue"> Example - 5 (Transformation)</h4>  
'''))  

# ##### Replace the sales values in the form of thousands

# Context: Some time you might want to modify columns to make them more readable. For instance, the sales column in the given data set has six digits, followed by two decimal places. You might want to make it more readable. You can convert the actual sales number to a number in thousands and make it a round figure. 

# eg. 300000 - 300K

# You can use the .floordiv function to achieve the transformation explained above. You can read more about the .floordiv method [here](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.floordiv.html). 

sales.Sales = sales.Sales.floordiv(1000)

sales.head()

sales.rename(columns={'Sales': 'Sales in Thousands'}, inplace=True)
sales.head()

##### Replace values in Profit percent of total

sales.head()

#sales['Profit']
total_sum = sales.Profit.sum()
sales['Profit % of Total'] = sales.Profit.apply(lambda x: x/total_sum*100)

sales.head()

##### Replace negative Profits with NAN

sales.loc[sales['Profit']<0, 'Profit'] = np.nan
sales.head()


# <a id='Pandas_Udemy'>Pandas_Udemy</a><br><br>

# In[36]:


import numpy as np
import pandas as pd

labels = ['a','b','c']
my_data = [10,20,30]
arr = np.array(my_data)

d = dict()
for i in range(len(labels)):
    d[labels[i]] = my_data[i]
print(d)


# In[37]:


display(pd.Series(data = my_data))
display(pd.Series(arr, labels))  # i/p: ([10 20 30],['a','b','c'])   ;   o/p: labels,arr
display(pd.Series(labels, arr))  # o/p: arr, labels
display(pd.Series(d))
display(pd.Series(data = labels))
display(pd.Series(data = [sum,print,len]))

a = pd.Series([1,2,3,4],['-1','-2','-3','-4'])
display(a[0],a[1])
b = pd.Series([1,2,3,4],['-1','-5','-3','-4'])
print(a + b)
print(a - b)
print(a ** b)


# ##### Pandas - Dataframes

# In[38]:


from numpy.random import randn

np.random.seed(1)
df = pd.DataFrame(randn(5,4),['A','B','C','D','E'],['W','X','Y','Z'])

display(df)
display(type(df))
display(type(df['W']))  # column selection
display((df[0:1]))  #row selection
display(type(df[0:1]))

df['E'] = 2* df['W']
display(df)
df.drop("Z",axis=1,inplace=True)  # default -> axis=0 -> row ; axis=1 -> column
display(df)
display(df.loc['A'])
display(df.iloc[1])  # 2nd row

display(df.loc['A','Y'])  #row, column
#display(df.loc['Y','A'])  #error

display(df.loc[['A','A',"B"],['Y',"X","Y"]]) 
display(df.iloc[[1,2,2],[1,2,2]]) 


# ##### Pandas - Dataframes - 2

# In[39]:


display(df > 0)
display(df[df['W']>0][['X','Y']])
display(df[df['W']>0][df['X']<=0])
display(df[(df['W']>0)&(df['X']<=0)])
display(df[(df['W']>0)|(df['X']<=0)])


# In[40]:


df.reset_index(inplace=False)
#df.set_index('index',inplace=True)
display(df)


# ##### Pandas - Dataframes - 3

# In[41]:


outside = ['g1','g1','g1','g2','g2','g2']
inside = [1,2,3,1,2,3]
hier_index = list(zip(outside,inside))
hier_index = pd.MultiIndex.from_tuples(hier_index)


# In[42]:


df = pd.DataFrame(randn(6,2),hier_index,['A','B'])
display(df)
display(df.loc['g1'])
display(df.loc['g1'].loc[1])


# In[43]:


df.index.names = ['Groups','Num']
display(df)
display(df.loc['g2'])
display(df.xs('g2'))
display(df.xs(1,level="Num"))
display(df.xs(2,level="Num"))


# ##### Pandas - Missing data

# In[44]:


d = {
    'A':[1,2,np.nan],
    'B':[5,np.nan,np.nan],
    'C':[1,2,3]
}
df = pd.DataFrame(d)  #shift + tab to get info about commands
display(df)
display(df.dropna())  # row-wise by default (axis=0)
#display(df)
df.dropna(axis=1)
display(df.dropna(thresh=2))


# In[45]:


display(df.fillna('blank'))
display(df.fillna(value=df['A'].mean()))
display(df.fillna(value=df['A'].median()))
display(df.fillna(value=df['A'].mode()))


# In[46]:


data = {
    'Company':['google','google','microsoft','microsoft','rakuten','rakuten'],
    'Person':['Sam','Charlie','Amy','Vanessa','Carl','Sarah'],
    'Sales':[200,120,340,124,243,350]
}

df = pd.DataFrame(data)
display(df)

byComp = df.groupby('Company') # <pandas.core.groupby.generic.DataFrameGroupBy object at 0x1291184f0>
display(byComp.sum())
display(byComp.sum().loc['rakuten'])
display(df.groupby('Company').min())


# In[47]:


display(df.groupby('Company').describe())


# 

# In[48]:


from datetime import datetime
import pytz

# Define the IST timezone
ist = pytz.timezone('Asia/Kolkata')

# Get the current time in UTC
utc_now = datetime.now(pytz.utc)

# Convert the current time to IST
ist_now = utc_now.astimezone(ist)

# Print the current time in IST
print("Current Time in IST:", ist_now.strftime('%Y-%m-%d %H:%M:%S'))


# In[ ]:




