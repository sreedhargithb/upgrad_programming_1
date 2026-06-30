#!/usr/bin/env python
# coding: utf-8

# # Problem Statement
# 
# > In the telecommunication industry, customers tend to change operators if not provided with attractive schemes
# and offers. It is very important for any telecom operator to prevent the present customers from churning to other operators. 
# In this case study would be to build an ML model which can predict if the customer will churn or not in a particular month based on the past data
# 
# 
# # Problem data
# <br>
# <a href="https://www.kaggle.com/competitions/telecom-churn-case-study-hackathon-c41/overview">Competition link</a>
# </br>
# <br>
# <a href="https://cdn.upgrad.com/uploads/production/a1e63cc1-7b2a-4d87-886f-fcb90bcda68b/Upgrad+hackathon.pdf">Upgrad Hackathon details</a>
# </br>
# <br>
# <a href="https://www.kaggle.com/competitions/telecom-churn-case-study-hackathon-c41/data">Dataset</a>
# <br>
# <br>
# Please note that you need to submit only from one account on Kaggle and the team name should be: <br><b>Name_of_member1_Name_of_member2</b>
# 
# # Business Objective
# To reduce customer churn, telecom companies need to predict which customers are at high risk of churn. The given dataset contains customer-level informations for few consecutive months June, July & August they are encoded as 6,7 & 8. The business objective is to predict the cusotmer which will churn in next month by analyzing the dataset
# High Value Customers:
# 
# One of the primary goal is to identify high value customers which are more likely to churn, as most of the profit comes from high value customers.
# Customers which are likely to churn will starting decreasing rhe recharge amount and other facilities. To identify high value customers, total_rech_data can be calculated and total dataser can be filtered which are greater than 70th percentil of the data
# 
# 
# # Helpful links
# 1. https://medium.com/analytics-vidhya/a-quick-guide-on-missing-data-imputation-techniques-in-python-2020-5410f3df1c1e#:~:text=KNNImputer%20is%20a%20multivariate%20data%20imputation%20technique%20used,in%20the%20training%20set%2C%20either%20weighted%20or%20unweighted.
# 2. https://www.geeksforgeeks.org/missing-data-imputation-with-fancyimpute/
# 3. Instructor's solution: https://onedrive.live.com/?authkey=%21ACueoqO8VnPDJmE&id=1024ECB43D19158B%21246926&cid=1024ECB43D19158B
# 
# 

# In[ ]:




