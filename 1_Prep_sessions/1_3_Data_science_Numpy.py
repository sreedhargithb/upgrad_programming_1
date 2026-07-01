#!/usr/bin/env python
# coding: utf-8

# ## Contents:
# 
# ### Upgrad notebooks:
# 1. [Intro](#Intro)<br>
# 2. [Exercise_1](#Exercise_1)<br>
# 3. [Case_study_Cricket_tournament](#Case_study_Cricket_tournament)<br>
# 4. [Creating_numpy_arrays](#Creating_numpy_arrays)<br>
# 5. [Operations_on_NumPy_Arrays](#Operations_on_NumPy_Arrays)<br>
# 6. [Compare_Computation_Times_in_NumPy_and_Standard_Python_Lists](#Compare_Computation_Times_in_NumPy_and_Standard_Python_Lists)<br>
# 7. [Numpy_practice_Udemy](#Numpy_practice_Udemy)<br>
# 7. [Numpy_practice](#Numpy_practice)<br>

# ### Upgrad notebooks:
# <a id='Intro'>Intro</a>

# <h2 style = "color : Brown"> Case Study - Cricket Tournament </h2>
# 
# A panel wants to select players for an upcoming league match based on their fitness. Players from all significant cricket clubs have participated in a practice match, and their data is collected. Let us now explore NumPy features using the player's data.</font>
# 
# <h4 style = "color : Sky blue"> Example - 1</h4>  
# 
# #### Heights of the players is stored as a regular Python list: height_in. The height is expressed in inches. Can you make a numpy array out of it ?

# In[91]:


# Define list
heights = [74, 74, 72, 72, 73, 69, 69, 71, 76, 71, 73, 73, 74, 74, 69, 70, 73, 75, 78, 79, 76, 74, 76, 72, 71, 75, 77, 74, 73, 74, 78, 73, 75, 73, 75, 75, 74, 69, 71, 74, 73, 73, 76, 74, 74, 70, 72, 77, 74, 70, 73, 75, 76, 76, 78, 74, 74, 76, 77, 81, 78, 75, 77, 75, 76, 74, 72, 72, 75, 73, 73, 73, 70, 70, 70, 76, 68, 71, 72, 75, 75, 75, 75, 68, 74, 78, 71, 73, 76, 74, 74, 79, 75, 73, 76, 74, 74, 73, 72, 74, 73, 74, 72, 73, 69, 72, 73, 75, 75, 73, 72, 72, 76, 74, 72, 77, 74, 77, 75, 76, 80, 74, 74, 75, 78, 73, 73, 74, 75, 76, 71, 73, 74, 76, 76, 74, 73, 74, 70, 72, 73, 73, 73, 73, 71, 74, 74, 72, 74, 71, 74, 73, 75, 75, 79, 73, 75, 76, 74, 76, 78, 74, 76, 72, 74, 76, 74, 75, 78, 75, 72, 74, 72, 74, 70, 71, 70, 75, 71, 71, 73, 72, 71, 73, 72, 75, 74, 74, 75, 73, 77, 73, 76, 75, 74, 76, 75, 73, 71, 76, 75, 72, 71, 77, 73, 74, 71, 72, 74, 75, 73, 72, 75, 75, 74, 72, 74, 71, 70, 74, 77, 77, 75, 75, 78, 75, 76, 73, 75, 75, 79, 77, 76, 71, 75, 74, 69, 71, 76, 72, 72, 70, 72, 73, 71, 72, 71, 73, 72, 73, 74, 74, 72, 75, 74, 74, 77, 75, 73, 72, 71, 74, 77, 75, 75, 75, 78, 78, 74, 76, 78, 76, 70, 72, 80, 74, 74, 71, 70, 72, 71, 74, 71, 72, 71, 74, 69, 76, 75, 75, 76, 73, 76, 73, 77, 73, 72, 72, 77, 77, 71, 74, 74, 73, 78, 75, 73, 70, 74, 72, 73, 73, 75, 75, 74, 76, 73, 74, 75, 75, 72, 73, 73, 72, 74, 78, 76, 73, 74, 75, 70, 75, 71, 72, 78, 75, 73, 73, 71, 75, 77, 72, 69, 73, 74, 72, 70, 75, 70, 72, 72, 74, 73, 74, 76, 75, 80, 72, 75, 73, 74, 74, 73, 75, 75, 71, 73, 75, 74, 74, 72, 74, 74, 74, 73, 76, 75, 72, 73, 73, 73, 72, 72, 72, 72, 71, 75, 75, 74, 73, 75, 79, 74, 76, 73, 74, 74, 72, 74, 74, 75, 78, 74, 74, 74, 77, 70, 73, 74, 73, 71, 75, 71, 72, 77, 74, 70, 77, 73, 72, 76, 71, 76, 78, 75, 73, 78, 74, 79, 75, 76, 72, 75, 75, 70, 72, 70, 74, 71, 76, 73, 76, 71, 69, 72, 72, 69, 73, 69, 73, 74, 74, 72, 71, 72, 72, 76, 76, 76, 74, 76, 75, 71, 72, 71, 73, 75, 76, 75, 71, 75, 74, 72, 73, 73, 73, 73, 76, 72, 76, 73, 73, 73, 75, 75, 77, 73, 72, 75, 70, 74, 72, 80, 71, 71, 74, 74, 73, 75, 76, 73, 77, 72, 73, 77, 76, 71, 75, 73, 74, 77, 71, 72, 73, 69, 73, 70, 74, 76, 73, 73, 75, 73, 79, 74, 73, 74, 77, 75, 74, 73, 77, 73, 77, 74, 74, 73, 77, 74, 77, 75, 77, 75, 71, 74, 70, 79, 72, 72, 70, 74, 74, 72, 73, 72, 74, 74, 76, 82, 74, 74, 70, 73, 73, 74, 77, 72, 76, 73, 73, 72, 74, 74, 71, 72, 75, 74, 74, 77, 70, 71, 73, 76, 71, 75, 74, 72, 76, 79, 76, 73, 76, 78, 75, 76, 72, 72, 73, 73, 75, 71, 76, 70, 75, 74, 75, 73, 71, 71, 72, 73, 73, 72, 69, 73, 78, 71, 73, 75, 76, 70, 74, 77, 75, 79, 72, 77, 73, 75, 75, 75, 73, 73, 76, 77, 75, 70, 71, 71, 75, 74, 69, 70, 75, 72, 75, 73, 72, 72, 72, 76, 75, 74, 69, 73, 72, 72, 75, 77, 76, 80, 77, 76, 79, 71, 75, 73, 76, 77, 73, 76, 70, 75, 73, 75, 70, 69, 71, 72, 72, 73, 70, 70, 73, 76, 75, 72, 73, 79, 71, 72, 74, 74, 74, 72, 76, 76, 72, 72, 71, 72, 72, 70, 77, 74, 72, 76, 71, 76, 71, 73, 70, 73, 73, 72, 71, 71, 71, 72, 72, 74, 74, 74, 71, 72, 75, 72, 71, 72, 72, 72, 72, 74, 74, 77, 75, 73, 75, 73, 76, 72, 77, 75, 72, 71, 71, 75, 72, 73, 73, 71, 70, 75, 71, 76, 73, 68, 71, 72, 74, 77, 72, 76, 78, 81, 72, 73, 76, 72, 72, 74, 76, 73, 76, 75, 70, 71, 74, 72, 73, 76, 76, 73, 71, 68, 71, 71, 74, 77, 69, 72, 76, 75, 76, 75, 76, 72, 74, 76, 74, 72, 75, 78, 77, 70, 72, 79, 74, 71, 68, 77, 75, 71, 72, 70, 72, 72, 73, 72, 74, 72, 72, 75, 72, 73, 74, 72, 78, 75, 72, 74, 75, 75, 76, 74, 74, 73, 74, 71, 74, 75, 76, 74, 76, 76, 73, 75, 75, 74, 68, 72, 75, 71, 70, 72, 73, 72, 75, 74, 70, 76, 71, 82, 72, 73, 74, 71, 75, 77, 72, 74, 72, 73, 78, 77, 73, 73, 73, 73, 73, 76, 75, 70, 73, 72, 73, 75, 74, 73, 73, 76, 73, 75, 70, 77, 72, 77, 74, 75, 75, 75, 75, 72, 74, 71, 76, 71, 75, 76, 83, 75, 74, 76, 72, 72, 75, 75, 72, 77, 73, 72, 70, 74, 72, 74, 72, 71, 70, 71, 76, 74, 76, 74, 74, 74, 75, 75, 71, 71, 74, 77, 71, 74, 75, 77, 76, 74, 76, 72, 71, 72, 75, 73, 68, 72, 69, 73, 73, 75, 70, 70, 74, 75, 74, 74, 73, 74, 75, 77, 73, 74, 76, 74, 75, 73, 76, 78, 75, 73, 77, 74, 72, 74, 72, 71, 73, 75, 73, 67, 67, 76, 74, 73, 70, 75, 70, 72, 77, 79, 78, 74, 75, 75, 78, 76, 75, 69, 75, 72, 75, 73, 74, 75, 75, 73]


# In[92]:


import numpy as np

heights_in = np.array(heights)


# In[93]:


print(heights_in)


# In[94]:


print(type(heights_in))


# <h4 style = "color : Sky blue"> Example - 2</h4>
# 
# #### Count the number of pariticipants

# In[95]:


print(len(heights))


# In[96]:


print(np.array(heights).shape)


# <h4 style = "color : Sky blue"> Example - 3</h4>
# 
# ####  Convert the heights from inches to meters

# In[97]:


heights = np.array(heights)
heights_m = heights * 0.0254

print(heights_m)


# <h4 style = "color : Sky blue"> Example - 4</h4>
# 
# #### A list of weights (in lbs) of the players is provided. Convert it to kg and calculate BMI

# In[98]:


weights_lb = [180, 215, 210, 210, 188, 176, 209, 200, 231, 180, 188, 180, 185, 160, 180, 185, 189, 185, 219, 230, 205, 230, 195, 180, 192, 225, 203, 195, 182, 188, 200, 180, 200, 200, 245, 240, 215, 185, 175, 199, 200, 215, 200, 205, 206, 186, 188, 220, 210, 195, 200, 200, 212, 224, 210, 205, 220, 195, 200, 260, 228, 270, 200, 210, 190, 220, 180, 205, 210, 220, 211, 200, 180, 190, 170, 230, 155, 185, 185, 200, 225, 225, 220, 160, 205, 235, 250, 210, 190, 160, 200, 205, 222, 195, 205, 220, 220, 170, 185, 195, 220, 230, 180, 220, 180, 180, 170, 210, 215, 200, 213, 180, 192, 235, 185, 235, 210, 222, 210, 230, 220, 180, 190, 200, 210, 194, 180, 190, 240, 200, 198, 200, 195, 210, 220, 190, 210, 225, 180, 185, 170, 185, 185, 180, 178, 175, 200, 204, 211, 190, 210, 190, 190, 185, 290, 175, 185, 200, 220, 170, 220, 190, 220, 205, 200, 250, 225, 215, 210, 215, 195, 200, 194, 220, 180, 180, 170, 195, 180, 170, 206, 205, 200, 225, 201, 225, 233, 180, 225, 180, 220, 180, 237, 215, 190, 235, 190, 180, 165, 195, 200, 190, 190, 185, 185, 205, 190, 205, 206, 220, 208, 170, 195, 210, 190, 211, 230, 170, 185, 185, 241, 225, 210, 175, 230, 200, 215, 198, 226, 278, 215, 230, 240, 184, 219, 170, 218, 190, 225, 220, 176, 190, 197, 204, 167, 180, 195, 220, 215, 185, 190, 205, 205, 200, 210, 215, 200, 205, 211, 190, 208, 200, 210, 232, 230, 210, 220, 210, 202, 212, 225, 170, 190, 200, 237, 220, 170, 193, 190, 150, 220, 200, 190, 185, 185, 200, 172, 220, 225, 190, 195, 219, 190, 197, 200, 195, 210, 177, 220, 235, 180, 195, 195, 190, 230, 190, 200, 190, 190, 200, 200, 184, 200, 180, 219, 187, 200, 220, 205, 190, 170, 160, 215, 175, 205, 200, 214, 200, 190, 180, 205, 220, 190, 215, 235, 191, 200, 181, 200, 210, 240, 185, 165, 190, 185, 175, 155, 210, 170, 175, 220, 210, 205, 200, 205, 195, 240, 150, 200, 215, 202, 200, 190, 205, 190, 160, 215, 185, 200, 190, 210, 185, 220, 190, 202, 205, 220, 175, 160, 190, 200, 229, 206, 220, 180, 195, 175, 188, 230, 190, 200, 190, 219, 235, 180, 180, 180, 200, 234, 185, 220, 223, 200, 210, 200, 210, 190, 177, 227, 180, 195, 199, 175, 185, 240, 210, 180, 194, 225, 180, 205, 193, 230, 230, 220, 200, 249, 190, 208, 245, 250, 160, 192, 220, 170, 197, 155, 190, 200, 220, 210, 228, 190, 160, 184, 180, 180, 200, 176, 160, 222, 211, 195, 200, 175, 206, 240, 185, 260, 185, 221, 205, 200, 170, 201, 205, 185, 205, 245, 220, 210, 220, 185, 175, 170, 180, 200, 210, 175, 220, 206, 180, 210, 195, 200, 200, 164, 180, 220, 195, 205, 170, 240, 210, 195, 200, 205, 192, 190, 170, 240, 200, 205, 175, 250, 220, 224, 210, 195, 180, 245, 175, 180, 215, 175, 180, 195, 230, 230, 205, 215, 195, 180, 205, 180, 190, 180, 190, 190, 220, 210, 255, 190, 230, 200, 205, 210, 225, 215, 220, 205, 200, 220, 197, 225, 187, 245, 185, 185, 175, 200, 180, 188, 225, 200, 210, 245, 213, 231, 165, 228, 210, 250, 191, 190, 200, 215, 254, 232, 180, 215, 220, 180, 200, 170, 195, 210, 200, 220, 165, 180, 200, 200, 170, 224, 220, 180, 198, 240, 239, 185, 210, 220, 200, 195, 220, 230, 170, 220, 230, 165, 205, 192, 210, 205, 200, 210, 185, 195, 202, 205, 195, 180, 200, 185, 240, 185, 220, 205, 205, 180, 201, 190, 208, 240, 180, 230, 195, 215, 190, 195, 215, 215, 220, 220, 230, 195, 190, 195, 209, 204, 170, 185, 205, 175, 210, 190, 180, 180, 160, 235, 200, 210, 180, 190, 197, 203, 205, 170, 200, 250, 200, 220, 200, 190, 170, 190, 220, 215, 206, 215, 185, 235, 188, 230, 195, 168, 190, 160, 200, 200, 189, 180, 190, 200, 220, 187, 240, 190, 180, 185, 210, 220, 219, 190, 193, 175, 180, 215, 210, 200, 190, 185, 220, 170, 195, 205, 195, 210, 190, 190, 180, 220, 190, 186, 185, 190, 180, 190, 170, 210, 240, 220, 180, 210, 210, 195, 160, 180, 205, 200, 185, 245, 190, 210, 200, 200, 222, 215, 240, 170, 220, 156, 190, 202, 221, 200, 190, 210, 190, 200, 165, 190, 185, 230, 208, 209, 175, 180, 200, 205, 200, 250, 210, 230, 244, 202, 240, 200, 215, 177, 210, 170, 215, 217, 198, 200, 220, 170, 200, 230, 231, 183, 192, 167, 190, 180, 180, 215, 160, 205, 223, 175, 170, 190, 240, 175, 230, 223, 196, 167, 195, 190, 250, 190, 190, 190, 170, 160, 150, 225, 220, 209, 210, 176, 260, 195, 190, 184, 180, 195, 195, 219, 225, 212, 202, 185, 200, 209, 200, 195, 228, 210, 190, 212, 190, 218, 220, 190, 235, 210, 200, 188, 210, 235, 188, 215, 216, 220, 180, 185, 200, 210, 220, 185, 231, 210, 195, 200, 205, 200, 190, 250, 185, 180, 170, 180, 208, 235, 215, 244, 220, 185, 230, 190, 200, 180, 190, 196, 180, 230, 224, 160, 178, 205, 185, 210, 180, 190, 200, 257, 190, 220, 165, 205, 200, 208, 185, 215, 170, 235, 210, 170, 180, 170, 190, 150, 230, 203, 260, 246, 186, 210, 198, 210, 215, 180, 200, 245, 200, 192, 192, 200, 192, 205, 190, 186, 170, 197, 219, 200, 220, 207, 225, 207, 212, 225, 170, 190, 210, 230, 210, 200, 238, 234, 222, 200, 190, 170, 220, 223, 210, 215, 196, 175, 175, 189, 205, 210, 180, 180, 197, 220, 228, 190, 204, 165, 216, 220, 208, 210, 215, 195, 200, 215, 229, 240, 207, 205, 208, 185, 190, 170, 208, 225, 190, 225, 185, 180, 165, 240, 220, 212, 163, 215, 175, 205, 210, 205, 208, 215, 180, 200, 230, 211, 230, 190, 220, 180, 205, 190, 180, 205, 190, 195]


# In[99]:


# Converting weights in lbs to kg

weights_kg = np.array(weights_lb) * 0.453592

print(weights_kg)


# In[100]:


# Calculate the BMI: bmi

bmi = weights_kg / (heights_m ** 2)

print(bmi)


# <h4 style = "color : Sky blue"> Sub-Setting Arrays</h4>
# 
# ##### Fetch the first element from the bmi array

# In[101]:


print(bmi[0])


# ##### Fetch the last element from the bmi array

# In[102]:


print(bmi[-1])


# #####  Fetch the first 5 elements from the bmi array

# In[103]:


print(bmi[0:5])


# ##### Fetch the last 5 elements from the bmi array

# In[104]:


print(bmi[-5:])


# <h4 style = "color : Sky blue"> Conditional Sub-Setting Arrays</h4>
# 
# ##### Count the number of pariticipants who are underweight i.e. bmi < 21
# 

# In[105]:


bmi < 21


# In[106]:


print(bmi [ bmi<21])


# In[107]:


underweight_players = bmi [ bmi<21]
print(underweight_players)


# In[108]:


print(underweight_players.size)


# <h4 style = "color : Sky blue"> NumPy Functions</h4>

# ##### Find the largest BMI value

# In[109]:


print(max(bmi))


# In[110]:


print(bmi.max())


# ##### Find lowest BMI value

# In[111]:


print(bmi.min())


# ##### Find average BMI value

# In[112]:


print(bmi.mean())


# <a id='Exercise_1'>Exercise_1</a>

# In[113]:


# Exercise Background

# This small application based coding exercise is ment to expose you to the use of the numpy library as well as give you a taste of tasks that you might be needed to perform during machine learning.

# Usually, machine learning involves working on large data sets. This notebook will walk you through normalising the data and then dividing the data set into smaller subsets. It is recommended that while attempting each of the tasks visit the NumPy library to find the most appropriate function which can help you achieve the desired result. More often than not you will find the functions which you require prewritten in the library. The **numpy library** can be found [here.](https://numpy.org/doc/stable/)

# Without further ado, the first task is to mean normalise a data set. Mean normalising is a data transformation done to reduce the variations in the data set. For example, consider a data set which has integers between 0 and 10000. That is a lot of variation, and it becomes difficult to build ML algorithms on this data. So mean normalisation is done on such data, after the transformation, the mean of the data will be zero, and standard deviation will be 1.  Even though the actual values of data will change a lot, but the overall variation is still kept intact. If the concept of normalisation feels a bit unclear dont worry all of this will be covered in the future sections of this program. For now, let’s concentrate on the tasks at hand.


# Task 1: Mean Normalisation:

#**Question 1.1** Create a 2D of random integers between 0 and 10,000 (including both 0 and 10,000) with 25000 rows and 15 columns. This will be the dataset you will use in the notebook.

# import NumPy into Python
import numpy as np

# Create a 25000 x 15 ndarray with random integers in the interval [0, 10000].
X = np.random.randint(0,high = 10001, size = (25000,15))

# print the shape of X
print(X.shape)

# print the first row of X
print(X[1])

# Now that you created the array we will mean normalize it. The equation for normalisaing the data is given below:

# $\mbox{Norm_Col}_i = \frac{\mbox{Col}_i - \mu_i}{\sigma_i}$

# where $\mbox{Col}_i$ is the $i$th column of $X$, $\mu_i$ is average of the values in the $i$th column of $X$, and $\sigma_i$ is the standard deviation of the values in the $i$th column of $X$. To put it simply, to find the new value of each element, you have to subtract the mean of respective column form that value and divide the result with the standard deviation of that columns. Now the question is, Why are these operations being done column-wise? That is because usually all the procedures in ML are done column-wise. So it will be beneficial for us to develop the habit of thinking about data column-wise.

# **Question 1.2** Find the mean and the standard deviation of each of the columns in the dataset. The result will be two 1D arrays with 15 elements each, representing the mean and standard deviation for each of the columns in the dataset.

# Average of the values in each column of X
ave_cols = np.average(X, axis = 0)

# print ave_cols
print ("ave_cols: \n", ave_cols,"\n")

# Standard Deviation of the values in each column of X
std_cols = np.std(X, axis = 0)

# print std_cols
print ("std_cols: \n", std_cols,"\n")

#**Question 1.3** Print the shape of each both the arrays, they should have 15 elements each.

# Print the shape of ave_cols
print(ave_cols.shape)

# Print the shape of std_cols
print(std_cols.shape)

# **Question 1.4** Now that you have mean and standard deviation calculated, it is time to apply the transformation to the dataset.

# **HINT** The broadcast property of NumPy can make this a lot easier. You can read about it [here](https://numpy.org/doc/stable/user/basics.broadcasting.html).
# All you have to do is create one row of transformation values and repeat them through all the values.

# Mean normalize X
X_norm = (X - ave_cols)/std_cols

# **Explanation of the solution:**

# The shape X is (25000, 15), and the shape of  ave_cols is (15,). If their shapes are different, then how is it possible that the operation can be executed? It is because of the property of broadcasting. Broadcasting repeats the smaller arrays over and over until it becomes the same size as the larger array. In this case, it is straightforward to figure out how the repetition would have worked. The ave_cols will be repeated 25000 times, and the operation is executed.

# But be careful broadcasting also has some limitations, you can read about them in the link above.

# **Question 1.5** If the transformation has been performed correctly, the mean of elements in each column will be approximately 0. Also, the average of the **minimum** value in each column of X_norm and the average of the **maximum** value in each column of X_norm will have almost the same face value with opposite signs. Let’s confirm if the transformation has happened correctly.

# Print the average of all the values of X_norm
print (np.average(X_norm))

# Print the average of the minimum value in each column of X_norm
print (np.average(X_norm.min(axis = 0)))

# Print the average of the maximum value in each column of X_norm
print (np.average(X_norm.max(axis = 0)))

# Be mindful that the exact values might not match since the dataset was initialized using the random function.

# # Data Spliting

# After data processing, it is a regular practice in ML to split the dataset into three datasets.

# 1. A Training Set
# 2. A Cross Validation Set
# 3. A Test Set

# The ratios in which the data is split varies a bit from case to case. But the accepted standard 6:2:2 for train, test, and validation respectively. That is 60% for training data and so on. Again why is the data split or what is the signification of these smaller data sets? These questions are better left unanswered for now.
# The tanks assigned to you is to split the data in the given proportions randomly.
# For instance, if the data set had ten elements, this is how you would do it.

# # We create a random permutation of integers 0 to 9
# np.random.permutation(10)

# 1. training set = 8,3,7,5,2,6
# 2. Cross Validation Set = 1,9
# 3. Test Set = 0,4

# Notice, the propostion of the split has stayed the same, and the data points are randomply seleccted and data points do not repeat in different sets.

# **Question 2.1** Similarly, create a 1D array representing the indexes of the rows in the dataset X_norm. U can use the   `np.random.permutation()` function for randomising the indexes.

# Create a rank 1 ndarray that contains a random permutation of the row indices of `X_norm`
row_indices = np.random.permutation(np.arange(0,25000))

# Print the shape of row_indices
print(row_indices.shape)

#**Question 2.2** Split the row indexes in the needed proportions. You can use the slicing methods you have learnt in this session to make the job easier.

25000*0.6

25000*0.2

# Make any necessary calculations.
# You can save your calculations into variables to use later.
train = row_indices[:15000]
test = row_indices[15000:20000]
val  = row_indices[20000:]

#**Question 2.4** Now make use of the indexes that you made to split the data also similarly once the data is split print the shape of each of the smaller data sets. `X_train` should have 15000 rows and 15 columns. `X_test` should have 5000 rows and 15 columns. `X_val` should have 5000 rows and 15 columns.

# Create a Training Set
X_train = X_norm[train]

# Create a Cross Validation Set
X_Val = X_norm[test]

# Create a Test Set
X_test = X_norm[val]

# Print the shape of X_train
print(X_train.shape)

# Print the shape of X_crossVal
print(X_Val.shape)

# Print the shape of X_test
print(X_test.shape)


# <a id='Case_study_Cricket_tournament'>Case_study_Cricket_tournament</a>

# In[114]:


from IPython.display import display,HTML
display(HTML('''
<h2 style = "color : Brown"> Case Study - Cricket Tournament </h2>

<h4 style = "color : Sky blue"> Example - 1</h4>
'''))

##### Players list contain the height(inches) and weight(lbs) data for all the players

# list of height and weight of the players.
players = [(74, 180), (74, 215), (72, 210), (72, 210), (73, 188), (69, 176), (69, 209), (71, 200), (76, 231), (71, 180), (73, 188), (73, 180), (74, 185), (74, 160), (69, 180), (70, 185), (73, 189), (75, 185), (78, 219), (79, 230), (76, 205), (74, 230), (76, 195), (72, 180), (71, 192), (75, 225), (77, 203), (74, 195), (73, 182), (74, 188), (78, 200), (73, 180), (75, 200), (73, 200), (75, 245), (75, 240), (74, 215), (69, 185), (71, 175), (74, 199), (73, 200), (73, 215), (76, 200), (74, 205), (74, 206), (70, 186), (72, 188), (77, 220), (74, 210), (70, 195), (73, 200), (75, 200), (76, 212), (76, 224), (78, 210), (74, 205), (74, 220), (76, 195), (77, 200), (81, 260), (78, 228), (75, 270), (77, 200), (75, 210), (76, 190), (74, 220), (72, 180), (72, 205), (75, 210), (73, 220), (73, 211), (73, 200), (70, 180), (70, 190), (70, 170), (76, 230), (68, 155), (71, 185), (72, 185), (75, 200), (75, 225), (75, 225), (75, 220), (68, 160), (74, 205), (78, 235), (71, 250), (73, 210), (76, 190), (74, 160), (74, 200), (79, 205), (75, 222), (73, 195), (76, 205), (74, 220), (74, 220), (73, 170), (72, 185), (74, 195), (73, 220), (74, 230), (72, 180), (73, 220), (69, 180), (72, 180), (73, 170), (75, 210), (75, 215), (73, 200), (72, 213), (72, 180), (76, 192), (74, 235), (72, 185), (77, 235), (74, 210), (77, 222), (75, 210), (76, 230), (80, 220), (74, 180), (74, 190), (75, 200), (78, 210), (73, 194), (73, 180), (74, 190), (75, 240), (76, 200), (71, 198), (73, 200), (74, 195), (76, 210), (76, 220), (74, 190), (73, 210), (74, 225), (70, 180), (72, 185), (73, 170), (73, 185), (73, 185), (73, 180), (71, 178), (74, 175), (74, 200), (72, 204), (74, 211), (71, 190), (74, 210), (73, 190), (75, 190), (75, 185), (79, 290), (73, 175), (75, 185), (76, 200), (74, 220), (76, 170), (78, 220), (74, 190), (76, 220), (72, 205), (74, 200), (76, 250), (74, 225), (75, 215), (78, 210), (75, 215), (72, 195), (74, 200), (72, 194), (74, 220), (70, 180), (71, 180), (70, 170), (75, 195), (71, 180), (71, 170), (73, 206), (72, 205), (71, 200), (73, 225), (72, 201), (75, 225), (74, 233), (74, 180), (75, 225), (73, 180), (77, 220), (73, 180), (76, 237), (75, 215), (74, 190), (76, 235), (75, 190), (73, 180), (71, 165), (76, 195), (75, 200), (72, 190), (71, 190), (77, 185), (73, 185), (74, 205), (71, 190), (72, 205), (74, 206), (75, 220), (73, 208), (72, 170), (75, 195), (75, 210), (74, 190), (72, 211), (74, 230), (71, 170), (70, 185), (74, 185), (77, 241), (77, 225), (75, 210), (75, 175), (78, 230), (75, 200), (76, 215), (73, 198), (75, 226), (75, 278), (79, 215), (77, 230), (76, 240), (71, 184), (75, 219), (74, 170), (69, 218), (71, 190), (76, 225), (72, 220), (72, 176), (70, 190), (72, 197), (73, 204), (71, 167), (72, 180), (71, 195), (73, 220), (72, 215), (73, 185), (74, 190), (74, 205), (72, 205), (75, 200), (74, 210), (74, 215), (77, 200), (75, 205), (73, 211), (72, 190), (71, 208), (74, 200), (77, 210), (75, 232), (75, 230), (75, 210), (78, 220), (78, 210), (74, 202), (76, 212), (78, 225), (76, 170), (70, 190), (72, 200), (80, 237), (74, 220), (74, 170), (71, 193), (70, 190), (72, 150), (71, 220), (74, 200), (71, 190), (72, 185), (71, 185), (74, 200), (69, 172), (76, 220), (75, 225), (75, 190), (76, 195), (73, 219), (76, 190), (73, 197), (77, 200), (73, 195), (72, 210), (72, 177), (77, 220), (77, 235), (71, 180), (74, 195), (74, 195), (73, 190), (78, 230), (75, 190), (73, 200), (70, 190), (74, 190), (72, 200), (73, 200), (73, 184), (75, 200), (75, 180), (74, 219), (76, 187), (73, 200), (74, 220), (75, 205), (75, 190), (72, 170), (73, 160), (73, 215), (72, 175), (74, 205), (78, 200), (76, 214), (73, 200), (74, 190), (75, 180), (70, 205), (75, 220), (71, 190), (72, 215), (78, 235), (75, 191), (73, 200), (73, 181), (71, 200), (75, 210), (77, 240), (72, 185), (69, 165), (73, 190), (74, 185), (72, 175), (70, 155), (75, 210), (70, 170), (72, 175), (72, 220), (74, 210), (73, 205), (74, 200), (76, 205), (75, 195), (80, 240), (72, 150), (75, 200), (73, 215), (74, 202), (74, 200), (73, 190), (75, 205), (75, 190), (71, 160), (73, 215), (75, 185), (74, 200), (74, 190), (72, 210), (74, 185), (74, 220), (74, 190), (73, 202), (76, 205), (75, 220), (72, 175), (73, 160), (73, 190), (73, 200), (72, 229), (72, 206), (72, 220), (72, 180), (71, 195), (75, 175), (75, 188), (74, 230), (73, 190), (75, 200), (79, 190), (74, 219), (76, 235), (73, 180), (74, 180), (74, 180), (72, 200), (74, 234), (74, 185), (75, 220), (78, 223), (74, 200), (74, 210), (74, 200), (77, 210), (70, 190), (73, 177), (74, 227), (73, 180), (71, 195), (75, 199), (71, 175), (72, 185), (77, 240), (74, 210), (70, 180), (77, 194), (73, 225), (72, 180), (76, 205), (71, 193), (76, 230), (78, 230), (75, 220), (73, 200), (78, 249), (74, 190), (79, 208), (75, 245), (76, 250), (72, 160), (75, 192), (75, 220), (70, 170), (72, 197), (70, 155), (74, 190), (71, 200), (76, 220), (73, 210), (76, 228), (71, 190), (69, 160), (72, 184), (72, 180), (69, 180), (73, 200), (69, 176), (73, 160), (74, 222), (74, 211), (72, 195), (71, 200), (72, 175), (72, 206), (76, 240), (76, 185), (76, 260), (74, 185), (76, 221), (75, 205), (71, 200), (72, 170), (71, 201), (73, 205), (75, 185), (76, 205), (75, 245), (71, 220), (75, 210), (74, 220), (72, 185), (73, 175), (73, 170), (73, 180), (73, 200), (76, 210), (72, 175), (76, 220), (73, 206), (73, 180), (73, 210), (75, 195), (75, 200), (77, 200), (73, 164), (72, 180), (75, 220), (70, 195), (74, 205), (72, 170), (80, 240), (71, 210), (71, 195), (74, 200), (74, 205), (73, 192), (75, 190), (76, 170), (73, 240), (77, 200), (72, 205), (73, 175), (77, 250), (76, 220), (71, 224), (75, 210), (73, 195), (74, 180), (77, 245), (71, 175), (72, 180), (73, 215), (69, 175), (73, 180), (70, 195), (74, 230), (76, 230), (73, 205), (73, 215), (75, 195), (73, 180), (79, 205), (74, 180), (73, 190), (74, 180), (77, 190), (75, 190), (74, 220), (73, 210), (77, 255), (73, 190), (77, 230), (74, 200), (74, 205), (73, 210), (77, 225), (74, 215), (77, 220), (75, 205), (77, 200), (75, 220), (71, 197), (74, 225), (70, 187), (79, 245), (72, 185), (72, 185), (70, 175), (74, 200), (74, 180), (72, 188), (73, 225), (72, 200), (74, 210), (74, 245), (76, 213), (82, 231), (74, 165), (74, 228), (70, 210), (73, 250), (73, 191), (74, 190), (77, 200), (72, 215), (76, 254), (73, 232), (73, 180), (72, 215), (74, 220), (74, 180), (71, 200), (72, 170), (75, 195), (74, 210), (74, 200), (77, 220), (70, 165), (71, 180), (73, 200), (76, 200), (71, 170), (75, 224), (74, 220), (72, 180), (76, 198), (79, 240), (76, 239), (73, 185), (76, 210), (78, 220), (75, 200), (76, 195), (72, 220), (72, 230), (73, 170), (73, 220), (75, 230), (71, 165), (76, 205), (70, 192), (75, 210), (74, 205), (75, 200), (73, 210), (71, 185), (71, 195), (72, 202), (73, 205), (73, 195), (72, 180), (69, 200), (73, 185), (78, 240), (71, 185), (73, 220), (75, 205), (76, 205), (70, 180), (74, 201), (77, 190), (75, 208), (79, 240), (72, 180), (77, 230), (73, 195), (75, 215), (75, 190), (75, 195), (73, 215), (73, 215), (76, 220), (77, 220), (75, 230), (70, 195), (71, 190), (71, 195), (75, 209), (74, 204), (69, 170), (70, 185), (75, 205), (72, 175), (75, 210), (73, 190), (72, 180), (72, 180), (72, 160), (76, 235), (75, 200), (74, 210), (69, 180), (73, 190), (72, 197), (72, 203), (75, 205), (77, 170), (76, 200), (80, 250), (77, 200), (76, 220), (79, 200), (71, 190), (75, 170), (73, 190), (76, 220), (77, 215), (73, 206), (76, 215), (70, 185), (75, 235), (73, 188), (75, 230), (70, 195), (69, 168), (71, 190), (72, 160), (72, 200), (73, 200), (70, 189), (70, 180), (73, 190), (76, 200), (75, 220), (72, 187), (73, 240), (79, 190), (71, 180), (72, 185), (74, 210), (74, 220), (74, 219), (72, 190), (76, 193), (76, 175), (72, 180), (72, 215), (71, 210), (72, 200), (72, 190), (70, 185), (77, 220), (74, 170), (72, 195), (76, 205), (71, 195), (76, 210), (71, 190), (73, 190), (70, 180), (73, 220), (73, 190), (72, 186), (71, 185), (71, 190), (71, 180), (72, 190), (72, 170), (74, 210), (74, 240), (74, 220), (71, 180), (72, 210), (75, 210), (72, 195), (71, 160), (72, 180), (72, 205), (72, 200), (72, 185), (74, 245), (74, 190), (77, 210), (75, 200), (73, 200), (75, 222), (73, 215), (76, 240), (72, 170), (77, 220), (75, 156), (72, 190), (71, 202), (71, 221), (75, 200), (72, 190), (73, 210), (73, 190), (71, 200), (70, 165), (75, 190), (71, 185), (76, 230), (73, 208), (68, 209), (71, 175), (72, 180), (74, 200), (77, 205), (72, 200), (76, 250), (78, 210), (81, 230), (72, 244), (73, 202), (76, 240), (72, 200), (72, 215), (74, 177), (76, 210), (73, 170), (76, 215), (75, 217), (70, 198), (71, 200), (74, 220), (72, 170), (73, 200), (76, 230), (76, 231), (73, 183), (71, 192), (68, 167), (71, 190), (71, 180), (74, 180), (77, 215), (69, 160), (72, 205), (76, 223), (75, 175), (76, 170), (75, 190), (76, 240), (72, 175), (74, 230), (76, 223), (74, 196), (72, 167), (75, 195), (78, 190), (77, 250), (70, 190), (72, 190), (79, 190), (74, 170), (71, 160), (68, 150), (77, 225), (75, 220), (71, 209), (72, 210), (70, 176), (72, 260), (72, 195), (73, 190), (72, 184), (74, 180), (72, 195), (72, 195), (75, 219), (72, 225), (73, 212), (74, 202), (72, 185), (78, 200), (75, 209), (72, 200), (74, 195), (75, 228), (75, 210), (76, 190), (74, 212), (74, 190), (73, 218), (74, 220), (71, 190), (74, 235), (75, 210), (76, 200), (74, 188), (76, 210), (76, 235), (73, 188), (75, 215), (75, 216), (74, 220), (68, 180), (72, 185), (75, 200), (71, 210), (70, 220), (72, 185), (73, 231), (72, 210), (75, 195), (74, 200), (70, 205), (76, 200), (71, 190), (82, 250), (72, 185), (73, 180), (74, 170), (71, 180), (75, 208), (77, 235), (72, 215), (74, 244), (72, 220), (73, 185), (78, 230), (77, 190), (73, 200), (73, 180), (73, 190), (73, 196), (73, 180), (76, 230), (75, 224), (70, 160), (73, 178), (72, 205), (73, 185), (75, 210), (74, 180), (73, 190), (73, 200), (76, 257), (73, 190), (75, 220), (70, 165), (77, 205), (72, 200), (77, 208), (74, 185), (75, 215), (75, 170), (75, 235), (75, 210), (72, 170), (74, 180), (71, 170), (76, 190), (71, 150), (75, 230), (76, 203), (83, 260), (75, 246), (74, 186), (76, 210), (72, 198), (72, 210), (75, 215), (75, 180), (72, 200), (77, 245), (73, 200), (72, 192), (70, 192), (74, 200), (72, 192), (74, 205), (72, 190), (71, 186), (70, 170), (71, 197), (76, 219), (74, 200), (76, 220), (74, 207), (74, 225), (74, 207), (75, 212), (75, 225), (71, 170), (71, 190), (74, 210), (77, 230), (71, 210), (74, 200), (75, 238), (77, 234), (76, 222), (74, 200), (76, 190), (72, 170), (71, 220), (72, 223), (75, 210), (73, 215), (68, 196), (72, 175), (69, 175), (73, 189), (73, 205), (75, 210), (70, 180), (70, 180), (74, 197), (75, 220), (74, 228), (74, 190), (73, 204), (74, 165), (75, 216), (77, 220), (73, 208), (74, 210), (76, 215), (74, 195), (75, 200), (73, 215), (76, 229), (78, 240), (75, 207), (73, 205), (77, 208), (74, 185), (72, 190), (74, 170), (72, 208), (71, 225), (73, 190), (75, 225), (73, 185), (67, 180), (67, 165), (76, 240), (74, 220), (73, 212), (70, 163), (75, 215), (70, 175), (72, 205), (77, 210), (79, 205), (78, 208), (74, 215), (75, 180), (75, 200), (78, 230), (76, 211), (75, 230), (69, 190), (75, 220), (72, 180), (75, 205), (73, 190), (74, 180), (75, 205), (75, 190), (73, 195)]

print(len(players))

print(players[1][1])

import numpy as np

np_players = np.array(players)

print(np_players)

print(type(np_players))



display(HTML('''
<h4 style = "color : Sky blue"> Example - 2 (Numpy Attributes)</h4>
'''))

##### Print the structure of the 2-D Array

print(np_players.shape)

##### Print the dimensions of the array

print(np_players.ndim)

##### Print the data type of elements in the array

print(np_players.dtype)

##### Print the size of a single item of the array

print(np_players.itemsize)

display(HTML('''
<h4 style = "color : Sky blue"> Example - 3</h4>
'''))

##### Convert the heights to meters and weights to kg

players_converted = np_players * [0.0254, 0.453592]

print(players_converted)

display(HTML('''
<h4 style = "color : Sky blue"> Sub-Setting 2-D Arrays</h4>
'''))

##### Fetch the first row from the array

print(players_converted[0])

##### Fetch the first row 2nd element from the array

print(players_converted[0][1])

##### Fetch the first column from the array

print(players_converted[:, 0])

##### Fetch the height (1st column) of 125th player from the array

print(players_converted[124][0])

print(players_converted[124,0])

display(HTML('''
<h4 style = "color : Sky blue"> Conditional Sub-Setting Arrays</h4>
'''))

##### Fetch height and weight of players with height above 1.8m


tall_players = players_converted[players_converted[:,0] > 1.8]

print(players_converted.shape)

print(tall_players.shape)

##### Skills Array - holds the player key skills.

skills = np.array(['Keeper', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Bowler', 'Batsman', 'Batsman', 'Bowler', 'Bowler', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Batsman', 'Keeper', 'Bowler', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Keeper', 'Keeper', 'Keeper', 'Keeper', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Batsman', 'Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Batsman', 'Batsman', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Bowler', 'Batsman', 'Keeper', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Bowler', 'Keeper-Batsman', 'Keeper', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Keeper', 'Batsman', 'Bowler', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Bowler', 'Bowler', 'Bowler', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Keeper', 'Batsman', 'Keeper', 'Bowler', 'Bowler', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Bowler', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Keeper', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper', 'Bowler', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Batsman', 'Batsman', 'Bowler', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Batsman', 'Batsman', 'Batsman', 'Keeper', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Bowler', 'Bowler', 'Bowler', 'Batsman', 'Keeper', 'Bowler', 'Bowler', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Batsman', 'Bowler', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper', 'Keeper', 'Batsman', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper', 'Batsman', 'Bowler', 'Keeper', 'Keeper', 'Batsman', 'Bowler', 'Bowler', 'Batsman', 'Keeper', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Bowler', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Batsman', 'Bowler', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Bowler', 'Batsman', 'Batsman', 'Batsman', 'Keeper', 'Keeper', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Batsman', 'Batsman', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Bowler', 'Batsman', 'Keeper', 'Bowler', 'Keeper', 'Bowler', 'Batsman', 'Keeper', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Batsman', 'Keeper', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper', 'Batsman', 'Bowler', 'Keeper', 'Bowler', 'Batsman', 'Batsman', 'Bowler', 'Batsman', 'Batsman', 'Bowler', 'Bowler', 'Bowler', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Batsman', 'Bowler', 'Keeper', 'Bowler', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Batsman', 'Batsman', 'Keeper', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Bowler', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Bowler', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Bowler', 'Keeper', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Bowler', 'Batsman', 'Keeper', 'Keeper', 'Keeper-Batsman', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper', 'Batsman', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Batsman', 'Bowler', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Bowler', 'Bowler', 'Keeper', 'Bowler', 'Keeper', 'Batsman', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Bowler', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Batsman', 'Bowler', 'Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Batsman', 'Batsman', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Batsman', 'Batsman', 'Bowler', 'Batsman', 'Keeper', 'Bowler', 'Bowler', 'Keeper', 'Bowler', 'Bowler', 'Keeper', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Keeper-Batsman', 'Keeper-Batsman', 'Bowler', 'Bowler', 'Bowler', 'Batsman', 'Bowler', 'Keeper-Batsman', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper', 'Keeper-Batsman', 'Batsman', 'Bowler', 'Keeper-Batsman'])
print(skills)

##### Fetch Heights of the Batsmen

batsmen = players_converted[skills == 'Batsman']

print(batsmen.shape)

print(batsmen[:, 0])



# <a id='Creating_numpy_arrays'>Creating_numpy_arrays</a>

# In[115]:


display(HTML('''
<h2 style = "color : Brown"> Creating NumPy Arrays </h2 >
'''))

#  The following ways are commonly used when you know the size of the array beforehand:
# * ```np.ones()```: Create array of 1s
# * ```np.zeros()```: Create array of 0s
# * ```np.random.random()```: Create array of random numbers
# * ```np.arange()```: Create array with increments of a fixed step size
# * ```np.linspace()```: Create array of fixed length

import numpy as np

##### Tip: Use help to see the syntax when required

help(np.ones)

##### Creating a 1 D array of ones

arr = np.ones(5)
print(arr)

##### Notice that, by default, numpy creates data type = float64



print(arr.dtype)

##### Can provide dtype explicitly using dtype


arr = np.ones(5, dtype=int)
print(arr)

print(arr.dtype)

##### Creating a 5  x 3 array of ones


np.ones((5,3))

##### Creating array of zeros

np.zeros(5)


# convert the type into integer.
np.zeros(5, dtype=int)

# Create a list of integers range between 1 to 5.
print(list(range(1,5)))

np.arange(3)

np.arange(3.0)

##### Notice that 3 is included, 35 is not, as in standard python lists

# From 3 to 35 with a step of 2

np.arange(3,35,2)

##### Array of random numbers


np.random.randint(2, size=10)

np.random.randint(3,5, size=10)

##### 2D Array of random numbers


np.random.random([3,4])

###### Sometimes, you know the length of the array, not the step size

# Array of length 20 between 1 and 10

np.linspace(1,10,20)

display(HTML('''
<h2 style = "color : Sky blue"> Exercises </h2>
'''))



# Apart from the methods mentioned above, there are a few more NumPy functions that you can use to create special NumPy arrays:

# -  `np.full()`: Create a constant array of any number ‘n’
# -  `np.tile()`: Create a new array by repeating an existing array for a particular number of times
# -  `np.eye()`: Create an identity matrix of any dimension
# -  `np.random.randint()`: Create a random array of integers within a particular range



# <a id='Operations_on_NumPy_Arrays'>Operations_on_NumPy_Arrays</a>

# In[116]:


display(HTML('''
<h2 style = "color : Brown"> Operations on NumPy Arrays</h2>
'''))

# The learning objectives of this section are:

# * Manipulate arrays
#     * Reshape arrays
#     * Stack arrays
# * Perform operations on arrays
#     * Perform basic mathematical operations
#     * Apply built-in functions
#     * Apply your own functions
#     * Apply basic linear algebra operations


import numpy as np

display(HTML('''
<h4 style = "color : Sky blue"> Example - 1 (Arithmatric Operations)</h4>
'''))


array1 = np.array([10,20,30,40,50])
array2 = np.arange(5)

print(array1)

print(array2)

# Add array1 and array2.
array3 = array1 + array2

print(array3)

display(HTML('''
<h4 style = "color : Sky blue"> Example - 2</h4>
'''))

array4 = np.array([1,2,3,4])

#array4 + array1 will cause error due to size mismatch

print (array1.shape)

print (array4.shape)

display(HTML('''
<h4 style = "color : Sky blue"> Example - 3</h4>
'''))


array = np.linspace(1, 10, 5)
print(array)

array*2

array**2

display(HTML('''
<h4 style = "color : Sky blue"> Stacking Arrays</h4>
'''))


####  ```np.hstack()``` and ```n.vstack()```

#Stacking is done using the ```np.hstack()``` and ```np.vstack()``` methods. For horizontal stacking, the number of rows should be the same, while for vertical stacking, the number of columns should be the same.

# Note that np.hstack(a, b) throws an error - you need to pass the arrays as a list
a = np.array([1, 2, 3])
b = np.array([2, 3, 4])

np.hstack((a,b))

np.vstack((a,b))

np.arange(12)

np.arange(12).reshape(3,4)

array1 = np.arange(12).reshape(3,4) #3x4
array2 = np.arange(20).reshape(5,4) #5x4

print (array1, '\n', array2)

np.vstack((array1,array2))

display(HTML('''
<h4 style = "color : Sky blue"> Example - 4 (Numpy Built-in functions)</h4>
'''))


print(array1)

np.power(array1, 3)

np.arange(9).reshape(3,3)

x = np.array([-2,-1, 0, 1,2])
print(x)

print(abs(x))

np.absolute(x)

display(HTML('''
<h4 style = "color : Sky blue"> Example - 5 (Trignometric functions)</h4>
'''))

print(np.pi)

theta = np.linspace(0, np.pi, 5)

print(theta)

np.sin(theta)

np.cos(theta)

np.tan(theta)

display(HTML('''
<h4 style = "color : Sky blue"> Example - 6 (Exponential and logarithmic functions)</h4>
'''))

x = [1, 2, 3, 10]
x = np.array(x)

np.exp(x) # e=2.718...

# 2^1, 2^2, 2^3, 2^10
np.exp2(x)

np.power(x,3)

np.log(x)

np.log2(x)

np.log10(x)

print(np.log)

display(HTML('''
<h4 style = "color : Sky blue"> Example - 7</h4>
'''))


x = np.arange(5)
print(x)

y = x * 10
print(y)

y = np.empty(5)
print(y)

np.multiply(x, 12, out=y)

print(y)

y = np.zeros(10)
print(y)

np.power(2, x, out=y[::2])

print(y)

display(HTML('''
<h4 style = "color : Sky blue"> Example - 8 (Aggregates)</h4>
'''))


x = np.arange(1,6)
print(x)

print(sum(x))

np.add.reduce(x)

np.add.accumulate(x)

np.multiply.accumulate(x)



#### Apply Basic Linear Algebra Operations

# NumPy provides the ```np.linalg``` package to apply common linear algebra operations, such as:
# * ```np.linalg.inv```: Inverse of a matrix
# * ```np.linalg.det```: Determinant of a matrix
# * ```np.linalg.eig```: Eigenvalues and eigenvectors of a matrix

# Also, you can multiple matrices using ```np.dot(a, b)```.


# np.linalg documentation
help(np.linalg)

A = np.array([[6, 1, 1],
              [4, -2, 5],
              [2, 8, 7]])

print(A)

##### Rank of a matrix

np.linalg.matrix_rank(A)

##### Trace of matrix A

np.trace(A)

##### Determinant of a matrix

np.linalg.det(A)

##### Inverse of matrix A

print(A)

np.linalg.inv(A)

B = np.linalg.inv(A)

np.matmul(A,B) #actual matrix multiplication

A * B

##### Matrix A raised to power 3

np.linalg.matrix_power(A,3) # matrix multiplication A A A



# <a id='Compare_Computation_Times_in_NumPy_and_Standard_Python_Lists'>Compare_Computation_Times_in_NumPy_and_Standard_Python_Lists</a>

# In[117]:


### Compare Computation Times in NumPy and Standard Python Lists

# Now that we know how to use numpy, let us see code and witness the key advantages of numpy i.e. convenience and speed of computation.

# In the data science landscape, you'll often work with extremely large datasets, and thus it is important point for you to understand how much computation time (and memory) you can save using numpy, compared to standard python lists.

# Let's compare the computation times of arrays and lists for a simple task of calculating the element-wise product of numbers.

import time

## Comparing time taken for computation
list_1 = [i for i in range(1000000)]
list_2 = [j**2 for j in range(1000000)]

t0 = time.time()
product_list = list(map(lambda x, y: x*y, list_1, list_2))
t1 = time.time()
list_time = t1 - t0
print (t1-t0)

# numpy array
array_1 = np.array(list_1)
array_2 = np.array(list_2)

t0 = time.time()
product_numpy = array_1 * array_2
t1 = time.time()
numpy_time = t1 - t0
print (t1-t0)

print("The ratio of time taken is {}".format(list_time//numpy_time))



# In this case, numpy is **an order of magnitude faster** than lists. This is with arrays of size in millions, but you may work on much larger arrays of sizes in order of billions. Then, the difference is even larger.

# Some reasons for such difference in speed are:
# * NumPy is written in C, which is basically being executed behind the scenes
# * NumPy arrays are more compact than lists, i.e. they take much lesser storage space than lists


# The following discussions demonstrate the differences in speeds of NumPy and standard python:
# 1. https://stackoverflow.com/questions/8385602/why-are-numpy-arrays-so-fast
# 2. https://stackoverflow.com/questions/993984/why-numpy-instead-of-python-lists



# <a id='Numpy_practice_Udemy'>Numpy_practice_Udemy</a>

# In[118]:


'''
Links:
1. Questions: https://www.w3resource.com/python-exercises/numpy/index.php
2. Numpy reference: https://numpy.org/doc/stable/reference/
'''

import ast
import numpy as np
from IPython.display import display,HTML

import sys
sys.stdout.write("Hello")



# NumPy Basic [59 exercises with solution]

# 1
print(np.__version__)
    #print(np.show_config())

#2
    #print(np.info(np.add))

#3 to #8
li = [1,2,3,4,5]
arr = np.array(li)
print(np.array(li)[np.array(li)==0])
print(np.all(arr<4))
print(np.any(arr<4))
print(-np.inf>2)  # false
print(np.inf>2)  # true

li = np.array([1+1j, 1+0j, 4.5, 3, 2, 2j])
print(li)  # all elements as complex numbers
print(np.isreal(li))
print(np.iscomplex(li))
print(np.isscalar(li))

# 10
li1 = [1,2,3,4,50]
arr1 = np.array(li1)
print(np.greater(arr,arr1))
print(np.less(arr,arr1))
print(np.greater_equal(arr,arr1))
print(np.less_equal(arr,arr1))

#12
print("%d bytes" % (arr1.size * arr1.itemsize))

#13
print(np.zeros(5), np.ones(4), np.ones(4)*4)


#14
print(np.array([i for i in range(30,71) if(i%2==0)]))   #or
print(np.arange(30,71,2))

#16 , 17
# Numpy matrices: https://numpy.org/doc/stable/reference/generated/numpy.matrix.html
matrix = np.matrix('1 2;3 4')
print(np.identity(3))

print(np.random.randint(3, 500,1))
print(np.random.randn(3, 500))
print(np.random.rand(3, 500))
print(np.random.normal(3, 500))
print(np.random.rand(3, 500,1))

print(np.arange(1,30,2))  # range with intervals
print(np.linspace(1,30,2)) # range (equally distributed) with length

y = np.zeros((2, 3, 4,4))
print(y)
print(y.ndim)  # 4

print(np.identity(4))
print(np.eye(4))
print(np.identity(4)==np.eye(4))   # true

arr = np.random.randint(4,50,300)
print(arr.argmax())  # Returns the indices of the maximum values along an axis.
print(arr[arr.argmax()])
print(arr[arr.argmin()])
print(arr.dtype)
print(arr)
arr[:] = 99
print(arr)

# numpy array indexing

print(np.linspace(0,11))  #Its default value is 50
print(np.linspace(0,11,2))
arr = arr.reshape((30,10))
print(arr)  # product should be equal to the total size
print(arr[:23,:23])

# numpy operations

arr = np.array([1,2,3,4,5],dtype="float64")
print(arr + arr)
print(arr * arr)
print(arr **----- arr)  # if dtype is "int", then: ValueError: Integers to negative integer powers are not allowed.


print(np.exp(arr))

# numpy exercises

print(np.zeros(10) + np.ones(10))
print(np.ones(10)*5)
print(np.arange(8).reshape(2,4))

print(np.random.rand(1,2,1))  # dimensions
print(np.random.randn(25))

# numpy exercises

print(np.arange(1,101).reshape(10,10)/100)
#print(np.linspace(1,101).reshape(10,10)/100)  -> error



# <a id='Numpy_practice'>Numpy_practice</a>

# In[119]:


import numpy as np, pandas

#Classes overview (instance methods, class methods, static methods)
from IPython.display import display,HTML
display(HTML('''
<h3>Data Science</h3>
<h4>Numpy</h4>
<h4>Pandas</h4>
'''))


'''
Important numpy links and references:
1. https://numpy.org/doc/stable/reference/
'''

heights = [1,2,3,4,5,6]
#print(heights*2.54)   # TypeError: can't multiply sequence by non-int of type 'float'

print(np.array(heights)*2.54) # homogeneous
print(np.array(heights)**---------------------------------------------2.54)


# Numpy arithmetic operations

display(HTML('''
<h3>Numpy arithmetic operations</h3>

'''))

list_1 = [1,2,3,4]
list_2 = [7,8,9,0]

print(list_1 + list_2)
print(np.array(list_1) + np.array(list_2))
np_1 = np.array(list_1)

print(np_1*0.0254)
print(len(np_1))
print(np_1.size)
print(np_1.shape)
print(np_1/2)
print(np_1//2)
print(np_1**2)
print(np_1+2)
print(np_1-2)
print(np_1)

display(HTML("<h3>Trigonometric:</h3>"))
print(np.sin(np.cos(np.tan(np.sin(np_1)))))
print(1/np.sin(np_1))   # cosec


display(HTML("<h3>Median,mean,std</h3>"))
print(np.mean(list_1), np.median(list_1))
print(np.min(list_1), np.max(list_1))
print(np.std(list_1))

display(HTML('''
<h3>Indexing, splitting, slicing</h3>

'''))
print(np_1[0])
print(np_1[-0])
print(np_1[-----1])
print(np_1[---+---1])
print(np_1[-4:-1])
print(np_1[-4:])
print(np_1[::-2])

display(HTML('''
<h3>Conditional sub-setting arrays</h3>
'''))

print(np_1<3)
print(np_1[np_1<3])
print((np_1<3).size)

display(HTML('''
<h4>Multiple-Conditioned sub-setting arrays</h4>
'''))
print(np_1[np.where((np_1%2==0) & (np_1<4))])
print(np_1[np.where((np_1%2==0) | (np_1<4))])

display(HTML('''
<h3>Numpy functions</h3>
'''))

print(np_1)
print(sum(np_1))
print(max(np_1))
print(min(np_1))
print((np_1).min())
print((np_1).max())
print((np_1).mean())
print(type(np_1))

display(HTML('''
<h3>Multidimensional arrays</h3>
'''))

players = [[12,23],[23,34],[34,45],[45,56],[56,67],[67,78],[78,89]]
print(np.array(players))
players = [(12,23),(23,34),(34,45),(45,56),(56,67),(67,78),(78,89)]
print(np.array(players))
players0 = [{12,23},{23,34},{34,45},{45,56},{56,67},{67,78},{78,89}]
print(np.array(players0))

display(HTML('''
<h3>Multidimensional arrays (continued)</h3>
'''))


# below will show error
'''
Traceback (most recent call last):
  File "/Users/sreedhar.k/Desktop/test.py", line 4, in <module>
    print(np.array(players1))
          ^^^^^^^^^^^^^^^^^^
ValueError: setting an array element with a sequence. The requested array has an inhomogeneous shape after 1 dimensions. The detected shape was (7,) + inhomogeneous part.
'''
# players1 = [{12,23},{23,34},(34,45),(45,56),(56,67),(67,78),(78,89)]
# print(np.array(players1))

np_players = np.array(players)
print(np_players.ndim)  # dimension
print(np_players.dtype)
print(np_players.astype('int64').dtype)  # to change dtype
print(np_players.itemsize)
print(np_players)
np_players_64 = np_players.astype('int64')
print(np_players_64.itemsize)

print(np_players_64*[2,5])

print(np_players_64[:,:2])
print(np_players_64[:,:2]>22)
print(np_players_64[:,:2]>[22,32])


######################## START: Taking Input in Python  ########################
# # Extracting elements from an array

# #print(np.array(ast.literal_eval(input())))
# import ast, numpy as np
# input_array=np.array(ast.literal_eval(input()))
# m=int(input())
# n=int(input())
# print(input_array[np.where((input_array<n)&(input_array>m))])
######################## END: Taking Input in Python  ########################



# In[120]:


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

