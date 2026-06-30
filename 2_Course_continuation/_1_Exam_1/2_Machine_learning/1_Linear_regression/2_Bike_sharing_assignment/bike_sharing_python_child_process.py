#!/usr/bin/env python
# coding: utf-8

# In[1]:


# working

import subprocess
list_files_1 = subprocess.run(["python","-c",'''
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
from IPython.display import display,HTML
warnings.filterwarnings('ignore')
import os
bike_df=pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_1_Exam_1/2_Machine_learning/1_Linear_regression/2_Bike_sharing_assignment/day.csv")
display(bike_df)
print(bike_df.shape)
print(bike_df.dtypes)
display(bike_df.head(5))

bike_df.rename(columns={'instant':'rec_id','dteday':'datetime','yr':'year','mnth':'month','weathersit':'weather_condition',
                       'hum':'humidity','cnt':'total_count'},inplace=True)

display(bike_df.head(5))

#Type casting the datetime and numerical attributes to category

bike_df['datetime'] = pd.to_datetime(bike_df['datetime'], format='%d-%m-%Y')

bike_df['season']=bike_df.season.astype('category')
bike_df['year']=bike_df.year.astype('category')
bike_df['month']=bike_df.month.astype('category')
bike_df['holiday']=bike_df.holiday.astype('category')
bike_df['weekday']=bike_df.weekday.astype('category')
bike_df['workingday']=bike_df.workingday.astype('category')
bike_df['weather_condition']=bike_df.weather_condition.astype('category')

display(bike_df.describe())
print("Done")
'''],timeout=500)
print()
print("The exit code was: %d" % list_files_1.returncode)


# In[2]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

