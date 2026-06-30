#!/usr/bin/env python
# coding: utf-8

# ### Structure of the course
# <img src="_readme/GetImage.png" />
# 
# ### Agenda: 
#     Module 1: Linear Regression (LR) 
#         Intro to SLR (Simple LR) 
#             Intro to ML 
#             Regression line 
#             Best-fit line 
#         SLR in Python 
#             Assumptions of SLR 
#             Reading and understanding the data 
#             Hypothesis testing in LR 
#             Building a linear model 
#             Residual analysis and Predictions 
#             LR using SKLearn 
# 
#         MLR (Multi-Linear Regression) 
#             New considerations 
#             Multicollinearity 
#             Dealing with Categorical variables 
#             Model Assessment and Comparison 
#             Feature selection 
#        MLR in Python: 
#            Reading and Understanding data 
#            Data preparation 
#            Initial steps 
#            Building the model 
#            Residual analysis and Predictions 
#            Variable selection using RFE (Recursive Feature Elimination)
#            
# ### Linear Regression
# 1. SLR and MLR (Single LR and Multi LR):<br>
# SLR:  
# 
# GitHub - ContentUpgrad/Linear-Regression: 
# 
# https://github.com/ContentUpgrad/Linear-Regression 
# 
# Read this  (https://www.mathsisfun.com/equation_of_line.html ) link to revise the physical significance of an equation of a straight line, and also to understand how to find the slope and intercept of a straight line from its graph. 
# 
# Why does the test statistic for β1 follow a t-distribution instead of a normal distribution? (Hypothesis testing in linear regression part 2) 
# 
# The calculation of F-statistic is a complex task and is not required. Hence it is out of the scope of this course. But interested students can check out this link : https://en.wikipedia.org/wiki/F-test. 
# 
#  
# 
# MLR: 
# 
# Y=β0+β1X1+β2X2+...+βpXp+ϵ 
# 
# In the link below, you can understand more about the overfitting concept. 
# 
# Overfitting : https://elitedatascience.com/overfitting-in-machine-learning 
# 
# Partial Least Squares (PLS) : https://support.minitab.com/en-us/minitab/18/help-and-how-to/modeling-statistics/regression/supporting-topics/partial-least-squares-regression/what-is-partial-least-squares-regression/ 
# 
# 2. R-squared vs Adjusted R-squared:<br>
#  The major difference between R-squared and Adjusted R-squared is that R-squared doesn't penalise the model for having more number of variables. Thus, if you keep on adding variables to the model, the R-squared will always increase (or remain the same in the case when the value of correlation between that variable and the dependent variable is zero). Thus, R-squared assumes that any variable added to the model will increase the predictive power. 
# 
# Adjusted R-squared on the other hand, penalises models based on the number of variables present in it. So if you add a variable and the Adjusted R-squared drops, you can be certain that that variable is insignificant to the model and shouldn't be used. So in the case of multiple linear regression, you should always look at the adjusted R-squared value in order to keep redundant variables out from your regression model. 
# 
# 3. Additional Reading 
# 
# To know more about dummy variables (https://stats.idre.ucla.edu/other/mult-pkg/faq/general/faqwhat-is-dummy-coding/) 
# 
# Why it's necessary to create dummy variables (https://stats.stackexchange.com/questions/89533/convert-a-categorical-variable-to-a-numerical-variable-prior-to-regression) 
# 
# When to Normalise data and when to standardise? (https://stackoverflow.com/questions/32108179/linear-regression-normalization-vs-standardization) 
# 
# Various scaling techniques (https://en.wikipedia.org/wiki/Feature_scaling) 
#  
# 
# The following links provide a detail study on AIC and other parameters used in automatic feature selection : 
# 
# AIC : https://en.wikipedia.org/wiki/Akaike_information_criterion 
# 
# BIC : https://en.wikipedia.org/wiki/Bayesian_information_criterion 
# 
# Mallows' CP : https://en.wikipedia.org/wiki/Mallows%27s_Cp 
# 
#  
# ### Instructors notes link
#  
# https://github.com/ContentUpgrad/Linear-Regression
#            
# ### Assignment -2 (Linear regression (Bike sharing) assignment)
# 
# Linear regression assignment:
# https://www.dropbox.com/scl/fo/vcj45zteqhi0qr9pl1xr8/h?dl=0&rlkey=aglrkxuyir3dob9t7j3fjesql
# 
# https://www.kaggle.com/code/lakshmi25npathi/bike-rental-count-prediction-using-python/notebook
# 
# Shared by professor: 
# https://drive.google.com/drive/folders/1VjrHKtgjLzWk9W_ImvJnyuw37XCr-iQS
# 
# <br>Location of assignment: http://localhost:8889/tree/2_Course_continuation/2_Machine_learning/1_Linear_regression/2_Bike_sharing_assignment
# 
# 
# 
# ### Logistic regression
# - https://www.youtube.com/watch?v=yIYKR4sgzI8&list=PLblh5JKOoLUKxzEP5HA2d-Li7IJkHfXSe
# 1. 
# <img src="_readme/GetImage_1.png" />
# 2. 
# <img src="_readme/GetImage_2.png" />
# 3. 
# <img src="_readme/GetImage_3.png" />
# 4. Logistic function:
# <img src="_readme/GetImage_4.png" />
# 5. Odds and log odds:
# <img src="_readme/GetImage_5.png" />
# <img src="_readme/GetImage_6.png" />
# 6. Confusion matrix:
# <img src="_readme/GetImage_7.jpg" />
# 
# 
# 7. Shared by professor:
# https://drive.google.com/drive/folders/1KGdUhqSBOjEmOeKdkmNL5s328eagkSTP
# 
# 8. Notes:<br>
#     i. The formula for standardising a value in a dataset is given by: (X−μ)/ σ <br>
#     ii. For a variable to be insignificant, the p-value should be greater than 0.05. <br>
#     iii. <u>Likelihood function:</u>
# 
#     So, the best fitting combination of β0 and β1 will be the one which maximises the product: 
#     (1−P1)(1−P2)(1−P3)(1−P4)(1−P6)(P5)(P7)(P8)(P9)(P10) 
#     This product is called the likelihood function. It is the product of: <br>
#     [(1−Pi)(1−Pi)------ for all non-diabetics --------] * [(Pi)(Pi) -------- for all diabetics -------]
#     
#  

# In[ ]:




