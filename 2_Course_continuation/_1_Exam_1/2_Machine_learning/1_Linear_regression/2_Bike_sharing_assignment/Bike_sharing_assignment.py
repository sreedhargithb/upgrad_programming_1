#!/usr/bin/env python
# coding: utf-8

# ## Contents:
# 
# 1. [Importing required packages](#Importing_required_packages)<br>
# 2. [Reading the training data:](#Reading_the_training_data)<br>
#     (i) [Renaming columns for better interpretation](#Renaming_columns_for_better_interpretation)<br>
#     (ii) [Typecasting the datetime and numerical attributes to category](#Typecasting_the_datetime_and_numerical_attributes_to_category)<br>
# 3. [Missing value analysis](#Missing_value_analysis)<br>
# 4. [Attributes distributions and trends:](#Attributes_distributions_and_trends)<br>
#     (i). [Monthly distribution of counts](#[Monthly_distribution_of_counts)<br>
#     (ii). [Yearly wise distribution of counts](#Yearly_wise_distribution_of_counts)<br>
#     (iii). [Holiday wise distribution of counts](#Holiday_wise_distribution_of_counts)<br>
#     (iv). [Workingday-wise distribution of counts](#Workingday_wise_distribution_of_counts)<br>
#     (v). [Weather_condition_wise distribution of counts](#Weather_condition_wise_distribution_of_counts)<br>
# 5. [Outlier analysis: ](#Outlier_analysis)<br>
#     (i) [Total_count_outliers](#Total_count_outliers)<br>
#     (ii) [Temp_windspeed_humidity_outliers](#Temp_windspeed_humidity_outliers)<br>
#     (iii) [Replace and impute the outliers](#Replace_and_impute_the_outliers)<br>
#     (iv) [Replace the original dataset to imputated data](#Replace_the_original_dataset_to_imputated_data)<br>
# 6. [Normal Probability plot](#Normal_Probability_plot)
# 7. [Correlation matrix](#Correlation_matrix)
# 8. [Modelling the dataset:](#Modelling_the_dataset)<br>
#     (i) [Split the dataset into train and test in the ratio of 70:30](#Split_the_dataset_into_train_and_test_in_the_ratio_of_70_30)<br>
#     (ii) [Split the features into categorical and numerical features](#Split_the_features_into_categorical_and_numerical_features)<br>
#     (iii) [Decoding the training attributes](#Decoding_the_training_attributes)<br>
# 9. [Models covered:](#Models_covered)<br>
#     (i) [Linear Regression model](#Linear_Regression_model)<br>
# 10. [Linear Regression model](#Linear_Regression_model)<br>
#     <b><u>[Training model](#Training_model)</u><br></b>
#     (i)[Fit the training model](#Fit_the_training_model)<br>
#     (ii) [Accuracy of model](#Accuracy_of_model)<br>
#     (iii) [Cross validation prediction](#Cross_validation_prediction)<br>
#     (iv) [Cross validation prediction plot](#Cross_validation_prediction_plot)<br>
#     (v) [Model evalution metrics:-](#Model_evalution_metrics)    [R-squared and mean squared error score](#R_squared_and_mean_squared_error_score)<br>
#         
#      <b><u>[Testing model](#Testing_model)</u><br></b>
#     (vii) [Decoding the test attributes](#Decoding_the_test_attributes)<br>
#     (viii) [Model performance on test dataset (Predict the model)](#Model_performance_on_test_dataset)<br>
#     (x) [Model evaluation metrics:-](#Model_evaluation_metrics) [Root mean square error and mean absolute error scores](#Root_mean_square_error_and_mean_absolute_error_scores)<br>
#     (v) [Residual plot](#Residual_plot)<br>
# 11. [Observations and explanations](#Observations_and_explanations)   
# <hr>
# <hr>

#   <a id='Importing_required_packages'></a>
#   ### 1. Importing required packages

# In[1]:


# Note: I have used dark mode for Jupyter Notebook

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')
import os


# <a id='Reading_the_training_data'></a>
# ### 2. Reading the training data

# In[ ]:


bikes_df = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_1_Exam_1/2_Machine_learning/1_Linear_regression/2_Bike_sharing_assignment/day.csv")

display(bikes_df)


# In[ ]:


# shape of the dataset
bikes_df.shape


# In[ ]:


# data types
display(bikes_df.dtypes)


# In[5]:


display(bikes_df.head(10))


# <a id='Renaming_columns_for_better_interpretation'></a>
# #### (i) Renaming columns for better interpretation

# In[ ]:


bikes_df.rename(columns={
    'instant':'rec_id',
    "dteday":"datetime",
    "yr":"year",
    "mnth":"month",
    "weathersit":"weather_condition",
    "hum":"humidity",
    "cnt":"total_count"
},inplace=True)


# In[ ]:


display(bikes_df.head(5))


# <a id='Typecasting_the_datetime_and_numerical_attributes_to_category'></a>
# #### (ii) Typecasting the datetime and numerical attributes to category

# In[ ]:


# Convert datetime column to datetime type with specific format
bikes_df['datetime'] = pd.to_datetime(bikes_df['datetime'], format='%d-%m-%Y')

# Convert other columns to category type
bikes_df['season'] = bikes_df['season'].astype('category')
bikes_df['year'] = bikes_df['year'].astype('category')
bikes_df['month'] = bikes_df['month'].astype('category')
bikes_df['holiday'] = bikes_df['holiday'].astype('category')
bikes_df['weekday'] = bikes_df['weekday'].astype('category')
bikes_df['workingday'] = bikes_df['workingday'].astype('category')
bikes_df['weather_condition'] = bikes_df['weather_condition'].astype('category')


# #### Describe the dataframe

# In[9]:


display(bikes_df.describe())


# <a id='Missing_value_analysis'></a>
# ### 3. Missing value analysis
# 
# 
# #### No missing values present in training dataset

# In[ ]:


# missing values in dataset
display(bikes_df.isnull().sum())


# <a id='Attributes_distributions_and_trends'></a>
# ### 4. Attributes distributions and trends
# 
# #### (i). Monthly distribution of counts
# #### (ii). Yearly wise distribution of counts
# #### (iii). Holiday wise distribution of counts
# #### (iv). Workingday-wise distribution of counts
# #### (v). Weather_condition_wise distribution of counts
# <hr>

# <a id='Monthly_distribution_of_counts'></a>
# (i) Monthly distribution of counts

# In[11]:


fig,ax = plt.subplots(figsize=(16,8))
sns.set_style('dark') # style must be one of white, dark, whitegrid, darkgrid, ticks

# Barplot for seasonwise monthly distribution of counts

sns.barplot(x='month',y = 'total_count',data=bikes_df[['month','total_count','season']],ax=ax, hue='season')
ax.set_title('Seasonwise monthly distribution of counts')
plt.show()


# In[12]:


# Barplot for weekday-wise monthly distribution of counts

fig,ax1 = plt.subplots(figsize=(16,8))
sns.set_style('dark') 
# Barplot for seasonwise monthly distribution of counts

sns.barplot(data=bikes_df[['month','total_count','weekday']], x='month',y = 'total_count',ax=ax1, hue='weekday')
ax1.set_title('Weekday-wise monthly distribution of counts')
plt.show()


# From the above plots, we can observe that there is an increase in the bike rental count in spring and summer season , and then a decrease in the bike rental count in fall and winter season.
# 
# Here,
# Season 1 -> Spring
# Season 2 -> Summer
# Season 3 -> Fall
# Season 4 -> Winter
# 
# <hr>

# <a id='Yearly_wise_distribution_of_counts'></a>
# (ii) Yearwise distribution of counts

# In[ ]:


fig,ax = plt.subplots(figsize=(16,8))

# Violin for yearwise distribution of counts

sns.violinplot(data=bikes_df[['year','total_count']], x='year',y = 'total_count')
ax.set_title('Yearwise distribution of counts')
plt.show()


# From the violin plot, we can observe that the bike rental count distribution is higher in 2019 than in 2018.
# <hr>

# <a id='Holiday_wise_distribution_of_counts'></a>
# (iii). Holiday wise distribution of counts

# In[ ]:


fig,ax = plt.subplots(figsize=(16,8))

# Barplot for holiday distribution of counts
sns.set_style('dark') 
sns.barplot(hue='season', data = bikes_df, x='holiday',y = 'total_count')
ax.set_title('Holiday-wise distribution of counts')
plt.show()


# From the above bar plot, we can observe that during no holidays, the bike rental counts is the highest, compared to during holidays for different seasons.
# 
# Here, 0-> No holiday,
# 1-> Holiday
# <hr>

# <a id='Workingday_wise_distribution_of_counts'></a>
# (iv). Workingday-wise distribution of counts

# In[ ]:


fig,ax = plt.subplots(figsize=(16,8))

# Barplot for workingday distribution of counts
ax.set_title('Workingday-wise distribution of counts')
sns.barplot(data = bikes_df, hue='season', x='workingday',y = 'total_count')
plt.show()


# From the above bar plot, we can observe that there is no significant change in bike demand with working days and non working days.
# 
# Here, 0-> No working day,
# 1-> Working day
# <hr>

# <a id='Weather_condition_wise_distribution_of_counts'></a>
# (v). Weather_condition_wise distribution of counts

# In[16]:


fig,ax1 = plt.subplots(figsize=(16,8))

# Barplot for Weather_condition_wise distribution of counts
ax.set_title('Weather_condition_wise distribution of counts')
sns.barplot(ax=ax1, data = bikes_df[['month','total_count','weather_condition']], x='weather_condition',y = 'total_count')

plt.show()


# From the above bar plot, we can observe that during clear, partly cloudy weather, the bike rental count is the highest, second-highest during misty cloudy weather, and followed by 3rd highest, during light snow and light rain weather
# 
# <hr>
# <hr>

# <a id='Outlier_analysis'></a>
# ### 5. Outlier analysis

# <a id='Total_count_outliers'></a>
# #### (i) Total_count_outliers

# In[ ]:


fig,ax=plt.subplots(figsize=(16,8))
#Boxplot for total_count outliers
ax.set_title('total_count outliers')
sns.boxplot(data=bikes_df[['total_count']])

plt.show()


# From the box plot, we can observe that no outliers are present in total_count variable.
# <hr>

# <a id='Temp_windspeed_humidity_outliers'></a>
# #### (ii) Temp_windspeed_humidity_outliers

# In[ ]:


fig,ax=plt.subplots(figsize=(16,8))
#Box plot for Temp_windspeed_humidity_outliers
ax.set_title('Temp_windspeed_humidity_outiers')
sns.boxplot(data=bikes_df[['temp','windspeed','humidity']])

plt.show()


# From the box plot, we can observe that no outliers are present in normalized temp , but few outliers are present in normalized windspeed and humidity variables.
# <hr>

# <a id='Replace_and_impute_the_outliers'></a>
# #### (iii) Replace and impute the outliers
# 
# Data imputation is the substitution of estimated values for missing or inconsistent data items (fields).

# In[ ]:


#create dataframe for outliers
wind_hum=pd.DataFrame(bikes_df,columns=['windspeed','humidity'])
 #Cnames for outliers                     
cnames=['windspeed','humidity']       

for i in cnames:
    q75,q25=np.percentile(wind_hum.loc[:,i],[75,25]) # Divide data into 75%quantile and 25%quantile.
    iqr=q75-q25 #Inter quantile range
    max=q75+(iqr*1.5) #outer fence
    min=q25-(iqr*1.5) #inner fence
    wind_hum.loc[wind_hum.loc[:,i]<min,:i]=np.nan  #Replace with NA
    wind_hum.loc[wind_hum.loc[:,i]>max,:i]=np.nan  #Replace with NA
#Imputating the outliers by mean Imputation
wind_hum['humidity']=wind_hum['humidity'].fillna(wind_hum['humidity'].mean())
wind_hum['windspeed']=wind_hum['windspeed'].fillna(wind_hum['windspeed'].mean())


# <a id='Replace_the_original_dataset_to_imputated_data'></a>
# #### (iv) Replace the original dataset to imputated data

# In[20]:


#Replacing the imputated humidity
bikes_df['humidity']=bikes_df['humidity'].replace(wind_hum['humidity'])
#Replacing the imputated windspeed
bikes_df['windspeed']=bikes_df['windspeed'].replace(wind_hum['windspeed'])
bikes_df.head(5)


# <a id='Normal_Probability_plot'></a>
# ### 6. Normal Probability Plot
# 
# Normal probability plot is a graphical technique to identify substantive departures from normality and also it tells about goodness of fit.

# In[21]:


from scipy import stats
#Normal plot
fig=plt.figure(figsize=(16,8))
stats.probplot(bikes_df.total_count.tolist(),dist='norm',plot=plt)
plt.show()


# In the above probability plot, some target variable data points are deviating from normality.
# <hr>

# <a id='Correlation_matrix'></a>
# ### 7. Correlation matrix
# 
# Correlation matrix tells about linear relationship between attributes and helps us to build better models.
# 

# In[22]:


#Create the correlation matrix
correMtr=bikes_df[["temp","atemp","humidity","windspeed","casual","registered","total_count"]].corr()
mask=np.array(correMtr)
mask[np.tril_indices_from(mask)]=False
#Heat map for correlation matrix of attributes
fig,ax=plt.subplots(figsize=(16,8))
ax.set_title('Correlation matrix of attributes')
sns.heatmap(correMtr,mask=mask,vmax=0.8,square=True,annot=True,ax=ax)

plt.show()


# From the above correlation plot, we can observe that some features are positively correlated , and some are negatively correlated to each other. The temp and atemp are highly positively correlated to each other, it means that both are carrying same information.The total_count,casual and registered are highly positively correlated to each other. So, we are going to ignore atemp,casual and registered variable for further analysis.
# <hr>
# <hr>

# <a id='Modelling_the_dataset'></a>
# #### 8. Modelling the dataset

# In[23]:


#load the required libraries
from sklearn import preprocessing,metrics,linear_model
from sklearn.model_selection import cross_val_score,cross_val_predict,train_test_split


# <a id='Split_the_dataset_into_train_and_test_in_the_ratio_of_70_30'></a>
# (i) Split the dataset into train and test in the ratio of 70:30

# In[ ]:


#Split the dataset into the train and test data
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(bikes_df.iloc[:,0:-3],bikes_df.iloc[:,-1],test_size=0.3, random_state=42)

#Reset train index values
X_train.reset_index(inplace=True)
y_train=y_train.reset_index()

# Reset train index values
X_test.reset_index(inplace=True)
y_test=y_test.reset_index()

display(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
display(y_train.head())
display(y_test.head())


# <a id='Split_the_features_into_categorical_and_numerical_features'></a>
# (ii) Split the features into categorical and numerical features

# In[ ]:


#Create a new dataset for train attributes
train_attributes=X_train[['season','month','year','weekday','holiday','workingday','weather_condition','humidity','temp','windspeed']]
#Create a new dataset for test attributes
test_attributes=X_test[['season','month','year','weekday','holiday','workingday','humidity','temp','windspeed','weather_condition']]
#categorical attributes
cat_attributes=['season','holiday','workingday','weather_condition','year']
#numerical attributes
num_attributes=['temp','windspeed','humidity','month','weekday']


# <a id='Decoding_the_training_attributes'></a>
# (iii) Decoding the training attributes
# 

# In[26]:


#To get dummy variables to encode the categorical features to numeric. 
# A dummy variable is a binary variable that takes a value of 0 or 1. 
# One adds such variables to a regression model to represent factors which are of a binary nature 
# i.e. they are either observed or not observed.

train_encoded_attributes=pd.get_dummies(train_attributes,columns=cat_attributes)
print('Shape of transfomed dataframe::',train_encoded_attributes.shape)
train_encoded_attributes.head(5)


# <a id='Models_covered'></a>
# ### 9. Models covered in this assignment:
# (i) Linear Regression model<br>
# <hr>

# <a id='Linear_Regression_model'></a>
# ### 10. Linear Regression model

# <a id='Training_model'></a>
# <u>Training dataset</u>

# In[ ]:


#Training dataset for modelling
X_train=train_encoded_attributes
y_train=y_train.total_count.values


# <a id='Fit_the_training_model'></a>
# (i) Fit the training model

# In[28]:


#fit the trained model
lr_model=linear_model.LinearRegression()
lr_model
lr_model.fit(X_train,y_train)
lr_model


# <a id='Accuracy_of_model'></a>
# (ii) Accuracy of the model

# In[ ]:


#Accuracy of the model
lr=lr_model.score(X_train,y_train)
print('Accuracy of the model :',lr)
print('Model coefficients :',lr_model.coef_)
print('Model intercept value :',lr_model.intercept_)


# <a id='Cross_validation_prediction'></a>
# (iii) Cross validation prediction

# In[ ]:


#Cross validation prediction
predict=cross_val_predict(lr_model,X_train,y_train,cv=3)
predict


# <a id='Cross_validation_prediction_plot'></a>
# (iv) Cross validation prediction plot

# In[31]:


#Cross validation plot
fig,ax=plt.subplots(figsize=(16,8))
ax.set_title('Cross validation prediction plot')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
ax.scatter(y_train,y_train-predict)
ax.axhline(lw=2,color='red')

plt.show()


# Cross validation prediction plot tells about finite variance between actual target value and predicted target value. In this plot, some data points have same finite variance between them,  and some are not having it.

# <a id='Model_evalution_metrics'></a>
# (v) Model evalution metrics
# 
# <a id='R_squared_and_mean_squared_error_score'></a>
# (a) R-squared and mean squared error score

# In[ ]:


#R-squared scores
r2_scores = cross_val_score(lr_model, X_train, y_train, cv=3)
print('R-squared scores :',np.average(r2_scores))


# The R-squared or coefficient of determination is 0.81 on average , it means that predictor is only able to predict 81% of the variance in the target variable which is contributed by independent variables.

# <a id='Testing_model'></a>
# <u> Testing model</u>
# 
# <a id='Decoding_the_test_attributes'></a>
# (vii) Decoding the test attributes

# In[ ]:


#To get dummy variables to encode the categorical features to numeric
test_encoded_attributes=pd.get_dummies(test_attributes,columns=cat_attributes)
print('Shape of transformed dataframe :',test_encoded_attributes.shape)
test_encoded_attributes.head(5)


# In[ ]:


#Test dataset for prediction
X_test=test_encoded_attributes
y_test=y_test.total_count.values


# <a id='Model_performance_on_test_dataset'></a>
# (viii) Model performance on test dataset (Predict the model)

# In[ ]:


#predict the model
lr_pred=lr_model.predict(X_test)
display(lr_pred)


# <a id='Model_evaluation_metrics'></a>
# (v) Model evaluation metrics
# 
# <a id='Root_mean_square_error_and_mean_absolute_error_scores'></a>
# (a) Root mean square error and mean absolute error scores

# In[ ]:


import math
#Root mean square error 
rmse=math.sqrt(metrics.mean_squared_error(y_test,lr_pred))
#Mean absolute error
mae=metrics.mean_absolute_error(y_test,lr_pred)
print('Root mean square error :',rmse)
print('Mean absolute error :',mae)


# <a id='Residual_plot'></a>
# (v) Residual plot

# In[37]:


#Residual plot
fig, ax = plt.subplots(figsize=(16,8))
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text("Residual Plot")
ax.scatter(y_test, y_test-lr_pred)
ax.axhline(lw=2,color='black')

plt.show()


# Residual plot tells about finite variance between actual target value and predicted target value. In this plot,very less data points are having the same finite variance between them.
# <hr>
# <hr>

# <a id='Observations_and_explanations'></a>
# Observations and explanations:
# 
# 1. There is an increase in the bike rental count in spring and summer season , and then decrease in the bike rental count in fall and winter season.
# 
# 2. The bike rental count distribution is higher in 2019 than in 2018.
# 3. During no holidays, the bike rental counts is the highest, compared to during holidays for different seasons.
# 4. There is no significant change in bike demand with working days and non working days.
# 5. During clear, partly cloudy weather, the bike rental count is the highest, second-highest during misty cloudy weather, and followed by 3rd highest, during light snow and light rain weather.
# 6. Outlier analysis:<br>
#     (i) No outliers are present in total_count variable.<br>
#     (ii) No outliers are present in normalized temp but few outliers are present in normalized windspeed and humidity variables.
# 7. Normal probability plot is a graphical technique to identify substantive departures from normality and also it tells about goodness of fit. In our normal probability plot, some target variable data points are deviating from normality.
# 8. Correlation matrix tells about linear relationship between attributes and helps us to build better models. 
# From our correlation plot, we can observe that some features are positively correlated and some are negatively correlated to each other. The temp and atemp are highly positively correlated to each other, it means that both are carrying same information.The total_count,casual and registered are highly positively correlated to each other. So, we have ignored atemp,casual and registered variable for further analysis.
# 9. For modelling the datset, we split the dataset into train and test in the ratio of 70:30.
# 10. Training dataset:<br>
#     (i) While fitting Linear regression to our trained dataset, Accuracy of the model: 82.4 %<br>
#     (ii) Cross validation prediction plot tells about finite variance between actual target value and predicted target value. In our Cross validation prediction plot for training dataset, some data points are have same finite variance between them and some are not having it.<br>
#     (iii) Model Evaluation metrics: R-Squared (R² or the coefficient of determination) is a statistical measure in a regression model that determines the proportion of variance in the dependent variable that can be explained by the independent variable.The R-squared or coefficient of determination for our model is 0.81 on average , it means that predictor is only able to predict 81% of the variance in the target variable which is contributed by independent variables.<br>
# 11. Testing dataset:<br>
#     (i) Model Evaluation metrics: Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors), and The mean absolute error of a model with respect to a test set is the mean of the absolute values of the individual prediction errors on over all instances in the test set.<br>
#         For our model, 
#         Root mean square error : 802.4291866599553
#         Mean absolute error : 595.2441391283483
# 12. Residual plot tells about finite variance between actual target value and predicted target values. In our Residual plot,very less data points are having the same finite variance between them.

# In[38]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

