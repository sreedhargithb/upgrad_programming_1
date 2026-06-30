#!/usr/bin/env python
# coding: utf-8

# 1.....Description.......
# Generate a binomial distribution, tested 10 times, given the number of trials(n) and probability(p) of each trial.
# The input will contain seed, n and p in the same order.
# The output should contain a numpy array with 10 numbers representing the required binomial distribution.
# Hint: You can use numpy's random number generator here too. Remember to set the seed before generating numbers to ensure correct output.
# Sample Input:
# 0
# 10
# 0.5
# Sample Output:
# 
# [5 6 5 5 5 6 5 7 8 5]

# In[2]:


import numpy as np 
import pandas as pd 
pd.set_option('display.max_columns', 500)
seed=int(input())
n=int(input())
p=float(input())
np.random.seed(seed) # np.random.seed(seed) makes the random numbers predictable
s = np.random.binomial(n,p,10)
print(s)


# 2.....Description.....
# A card stack contains white and black cards. Two cards are drawn randomly without replacement. The probability of selecting a white and then a black card is x. The probability of selecting a white card in the first draw is y. You have to find the probability of drawing a black card, given that the first card drawn was white.
# The input will contain two lines with x and y respectively.  
# The output should be displayed as a float(no need to round it off).
# Sample Input:
# 0.2
# 0.5
# Sample Output:
# 0.4

# In[3]:


x=float(input())
y=float(input())
#write your code here
print(x/y)


# 3......Description......
# Consider that the scores of an entrance test are normally distributed. 
# If the test has a mean of m and a standard deviation of s.
# You have set the cut off as 90 marks for the test. What is the probability of a student passing the test?
# The input will contain m and s in two lines in the same order. The output has to be rounded off to two decimal places.
# 
# Please note that if the answer is 3.6, it is to be reported as 3.6 and not 3.60
# 
# Sample Input:
# 100
# 10
# Sample Output:
# 0.84

# In[4]:


import scipy.stats
m=float(input())#mean
sd=float(input())#standard deviation

print(round(scipy.stats.norm.sf(90,m,sd),2))


# ##### https://brainly.in/question/11866628
# 4.....Seating Arrangement.....
# Description
# N people sit around a circular table. You have to find the probability that two particular people won't be sitting together.
# 
# The input will have the number N and the output should have the probability printed as a float type number rounded off to four decimal places.
# 
# Hint: This problem follows the logic of circular permutation which you can read about here.
# 
# Please note that if the answer is 3.6, it is to be reported as 3.6 and not 3.6000
# 
# Sample Input:
# 15

# In[5]:


n=int(input())
print(round((n-3)/(n-1),4))


# In[6]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

