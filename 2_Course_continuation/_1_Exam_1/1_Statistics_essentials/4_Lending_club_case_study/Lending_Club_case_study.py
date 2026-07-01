#!/usr/bin/env python
# coding: utf-8

# # Lending Club Case Study

# # Problem Statement
# Working for a consumer finance company which specializes in lending various types of loans to urban customers. When the company receives a loan application, the company has to make a decision for loan approval based on the applicant’s profile. Two types of risks are associated with the bank’s decision:
# <br>
#     1) If the applicant is likely to repay the loan, then not approving the loan results in a loss of business to the company
# <br>
#     2) If the applicant is not likely to repay the loan, i.e. he/she is likely to default, then approving the loan may lead to a financial loss for the company
#     <br>
# <br>
# The data given to us contains information about past loan applicants and whether they ‘defaulted’ or not. 
# <br>
# The aim is to identify patterns which indicate if a person is likely to default, which may be used for taking actions such as denying the loan, reducing the amount of loan, lending (to risky applicants) at a higher interest rate, etc.
#  <br>
#  <br>
# If one is able to identify these risky loan applicants, then such loans can be reduced thereby cutting down the amount of credit loss. Identification of such applicants using EDA is the aim of this case study
# <br>
#  <br>
# Company wants to understand the driving factors (or driver variables) behind loan default, i.e. the variables which are strong indicators of default.  The company can utilize this knowledge for its portfolio and risk assessment.
# 

# # Steps Followed
# 1.	Import Required Libraries
# <BR>
# 2.	Data Overview<BR>
# 3.	Data Cleaning<BR>
# a.	Missing Data Treatment<BR>
# b.	Standardizing Values<BR>
# c.	Remove Irrelevant Variables<BR>
# d.	Outliers Analysis and Treatments<BR>
# e.	Derived Metrics & Binning<BR>
# 4.	Data Analysis<BR>
# a.	Univariate Analysis<BR>
# b.	Bivariate Analysis<BR>
# c.	Multivariate Analysis<BR>
# 5.	Insights<BR>
# 6.	Recommendations<BR>
# 
# 
# 

# In[1]:


#Importing the required libraries
import numpy as np #Math library
import pandas as pd #To work with dataset
import matplotlib.pyplot as plt #Graph library that use matplot in background
import seaborn as sns  #to plot some parameters in seaborn
import warnings  #to ignore warnings
warnings.filterwarnings('ignore')


# In[2]:


#Read the data from csv and load in the dataframe
lend_data=pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_1_Exam_1/1_Statistics_essentials/4_Lending_club_case_study/loan.csv")
print(lend_data.head())


# # Data Overview

# In[3]:


#Understand the list of variables and its info
print(lend_data.info(verbose=True,show_counts=True))


# In[4]:


#Find the number of rows and columns
print(lend_data.shape)


# # Data Cleaning

# In[5]:


## Find number of rows which has null values in all the columns

print("Number of rows which has null values in all the columns: ".format(lend_data.isnull().all(axis=1).sum()))


# There are no rows with all null values
# <br>
# <br>
# We hav 111 columns and 39717 rows.
# <br>
# 
# Find the number of attributes (columns) which has all null values and drop the same.

# In[6]:


#List the columns which has all null values and count the same
c=0
for i in lend_data.columns:
    if lend_data[i].isnull().sum()==lend_data.shape[0]:
        print(i)
        c+=1
print("Total number of columns which has all null values: ",c)


# We can drop all 54 columns as these will not add values for our analysis
# 

# In[7]:


#Drop columns with all null values
lend_data.dropna(axis=1, how="all", inplace=True)


# In[8]:


#Check the shape of data frame
print(lend_data.shape)


# Now we have 57 columns for the analysis.
# 

# Find the columns which has only one value

# In[9]:


c=0
for i in lend_data.columns:
    if lend_data[i].nunique()==1:#check the column only with one unique value
        print(i)
        c+=1
print("Total number of columns with only one value: ",c)


# Above 9 variables are having single value or in combination with NA.<br>
# Variables like collections_12_mths_ex_med, chargeoff_within_12_mths,tax_liens etc has value 0 or NA. As both values are not useful for our anlysis, we will drop these

# In[10]:


#Drop the columns in dataframe
lend_data.drop(['pymnt_plan', "initial_list_status",'collections_12_mths_ex_med','policy_code','application_type', 'acc_now_delinq', 'chargeoff_within_12_mths', 'delinq_amnt', 'tax_liens'], axis = 1, inplace = True)
print(lend_data.head())


# Identify the attributes which has only unique values across the data

# In[11]:


c=0
for i in lend_data.columns:
    if lend_data[i].nunique()==lend_data[i].shape[0]: #check columns only with unique values
        print(i)
        c+=1
print(c)


#  1) "id", "member_id","url": Remove unique value attributes. As these are unique values, it will not be much useful for our analysis. 
# <br>
# 2) "title", "emp_title","zip_code", "desc": These customer specific columns will also not contribute to our loan defaulting analysis and hence dropping the same
# <br>
# 3) Below attributes are specific to post loan approval process and hence we can drop it. delinq_2yrs,"last_credit_pull_d",mths_since_last_delinq,mths_since_last_record,revol_bal,out_prncp,total_pymnt,total_rec_prncp,total_rec_int,total_rec_late_fee,recoveries,collection_recovery_fee,last_pymnt_d,last_pymnt_amnt,next_pymnt_d,out_prncp_inv,total_pymnt_inv,funded_amnt.
# 
# 

# In[12]:


#Drop the columns in dataframe
lend_data.drop(["id", "member_id","url","title","emp_title", "zip_code","desc","delinq_2yrs","last_credit_pull_d","mths_since_last_delinq","mths_since_last_record","revol_bal","out_prncp","total_pymnt","total_rec_prncp","total_rec_int","total_rec_late_fee","recoveries","collection_recovery_fee","last_pymnt_d","last_pymnt_amnt","next_pymnt_d","out_prncp_inv","total_pymnt_inv","funded_amnt"], axis = 1, inplace = True)


# In[13]:


print(lend_data.info())


# #### Analyze pub_rec_bankruptcies Variable

# In[14]:


#Describe pub_rec_bankruptcies
print(lend_data['pub_rec_bankruptcies'].describe())


# In[15]:


#Draw the box plot to identify outliers
sns.boxplot(lend_data['pub_rec_bankruptcies'])
plt.title("Public Record Bankruptcies")
plt.show()


# As maximum values in "pub_rec_bankruptcies" attribute are 0, dropping this attribute.

# In[16]:


lend_data.drop(["pub_rec_bankruptcies"], axis = 1, inplace = True)


# In[17]:


print(lend_data.shape)


# Now we have 22 columns. Lets understand the features of each columns

# In[18]:


print(lend_data.info())
# lend_Data is the name of the dataframe


# We will analyze the object columns first. <br> 
# Finding value count for all object variables

# In[19]:


for i in lend_data.columns[lend_data.dtypes == 'object']:
    print(lend_data[i].value_counts())
    print('-----------------------------------------------')


# ## Data Standardizing Values

#  Term , int_rate,revol_util,emp_length has some strings and need to convert to numeric value
#  We can observe this from the above output

# In[20]:


#Remove months from the term and convert the type to int
lend_data["term"]=lend_data["term"].str.replace(" months","").astype("int")


# In[21]:


#Remove % from the interest rate and convert the type to float
lend_data["int_rate"]=lend_data["int_rate"].str.replace("%","").astype("float")


# In[22]:


#Remove % from the verifica and convert the type to float
lend_data["verifica"]=lend_data["verifica"].str.replace("%","").astype("float")


# In[23]:


#Find the % of null values in emplyment length variable
lend_data['emp_length'].isnull().sum()/lend_data['emp_length'].count()


# Only ~2.7% of data has null values and hence replace null value with mode value

# In[24]:


#Fill null value of emp_length with mode of emp_length
lend_data['emp_length'] = lend_data['emp_length'].fillna(lend_data['emp_length'].mode()[0])


# In[25]:


#emp_length column convert to int considering 0 means less than one year and 10 means ten or more years
lend_data['emp_length'] = lend_data['emp_length'].map(lambda x:x.split('+ years')[0] if x == '10+ years' else ('0' if x == '< 1 year' else(x.strip(' years')))).astype(int)


# In[26]:


#Finding ratio if null values for remaining columns

(lend_data.isna().sum()/len(lend_data.index))*100


# Only revol_util has null values. As this attribute is giving the credit utilization rate, it is important for our analysis. As we cant default with any value, we will remove the rows which has null value in this columns

# In[27]:


#Drop rows which has null verifica values
lend_data.dropna(axis = 0, subset = ["verifica"] , inplace = True)


# Compute month and year data for issue_d & earliest_cr_line attributes and store in new columns

# In[28]:


#convert the variable to date format
lend_data["issue_d"] = pd.to_datetime(lend_data['issue_d'],format='%b-%y')


# In[29]:


#Extract month and year from issue date variable and store in new variables
lend_data["issue_date_year"]=pd.DatetimeIndex(lend_data['issue_d']).year
lend_data["issue_date_month"]=pd.DatetimeIndex(lend_data['issue_d']).month


# In[30]:


#convert the variable to date format
lend_data["earliest_cr_line"] = pd.to_datetime(lend_data['earliest_cr_line'],format='%b-%y')


# In[31]:


#Extract month and year from earliest_cr_line variable and store in new variables
lend_data["earliest_cr_line_year"]=pd.DatetimeIndex(lend_data['earliest_cr_line']).year
lend_data["earliest_cr_line_month"]=pd.DatetimeIndex(lend_data['earliest_cr_line']).month


# In[32]:


print(lend_data.head())


# # Filtering Data

# As we are analysing on defaulters posibilities, we need to consider only "Fully Paid" and "Charges Off" load statuses 

# In[33]:


#Remove data with loan status as current
lend_data = lend_data[lend_data.loan_status != "Current"]


# # Data Analysis

# In[34]:


print(lend_data.shape)


# In[35]:


#Create list variables to store continuous and categorial variables separately
cont=['loan_amnt','funded_amnt_inv', 'term', 'int_rate', 'installment', 'emp_length', 'annual_inc', 'dti','inq_last_6mths', 'open_acc', 'pub_rec', 'verifica', 'total_acc']
catg= ['grade', 'sub_grade', 'home_ownership', 'verification_status', 'issue_d', 'loan_status', 'purpose', 'addr_state', 'earliest_cr_line']


# In[36]:


print(lend_data[cont])


# Lets analyze "annual_inc" attribute

# In[37]:


#Draw boxplot to find outliers
sns.boxplot(lend_data['annual_inc'])
plt.title("Annual Income")
plt.show()


# It seems we have outliers in annual income. lets understand the outliers with percentile functions

# In[38]:


#Print annual income value for below listed quantile
print(lend_data.annual_inc.quantile([0.1,0.15,0.2,0.25,0.5, 0.7,0.75,0.8,0.85,0.9, 0.95,0.96, 0.97,0.98, 0.99]))


# From above data, we can see more outliers are from 0.96 percentile. So we will consider only the data which has annual income of lesser or equal to 95 percentile

# In[39]:


#Remove the rows with annual income greater than 95 percentile
lend_data = lend_data[lend_data.annual_inc <= lend_data.annual_inc.quantile(0.95)]


# In[40]:


#Draw boxplot after outlier treatment
sns.boxplot(lend_data['annual_inc'])
plt.title("Annual Income")
plt.show()


# Lets create a new function to print graph and requried variables for analysis

# In[41]:


def univ_cont_anlys(cont_var,x_size=6,y_size=6,color_v="g",bin="auto"):
    plt.figure(figsize=(x_size,y_size))

    plt.boxplot(lend_data[cont_var]) #Draw box plot for given variable

    plt.title(cont_var)

    plt.xlabel(cont_var)

    plt.ylabel("Quartile Range")

    plt.show()

    plt.hist(lend_data[cont_var],color=color_v,bins=bin,edgecolor='blue') #Draw histogram for given variable

    plt.title(cont_var)

    plt.xlabel(cont_var)

    plt.ylabel("Frequency")

    plt.show()
    print(lend_data[cont_var].describe()) #describe the given variable




# In[42]:


# Univariate Analysis for annual_inc
univ_cont_anlys('annual_inc',color_v='b',bin=10)


# ##### Annual Income Variable Insights
# After cleaning outliers earlier, we can see maximum number of loan are uner 4th slab between 50000 and 60000

# In[43]:


# Univariate Analysis for Loan Amount
univ_cont_anlys('loan_amnt',color_v='b',bin=20)


# ##### Loan Amount Insights
# There is no much outliers with bigger difference and hence no cleanup required
# <br>
# Maximum loan applications are in 3rd bin of loan amount ie around 5000.

# In[44]:


# Univariate Analysis for Funded Amount Investor
univ_cont_anlys('funded_amnt_inv',color_v='b',bin=20)


# ##### Funded Amount and Funded Amount Investors Variables Insights
# There is no much outliers with bigger difference and hence no cleanup required
# <br>
# Maximum loan applications are in 3rd bin of funded loan investor amount ie around 5000.

# In[45]:


# Univariate Analysis for term
print(lend_data['term'].value_counts()) # Print count of each values
lend_data['term'].value_counts().plot.bar() #Draw bar plot based on value count
plt.title("Terms Value Distributions")
plt.xlabel("Terms")
plt.ylabel("Frequency")
plt.show()


# ##### Term Variable Insights
# We have only two terms 36 & 60
# <Br>
# Maximum number of loans are in 36 terms only

# In[ ]:





# In[46]:


# Univariate Analysis for installment
univ_cont_anlys('installment',color_v='b',bin=10)


# ##### Installment Variable Insights
# Maximum number of loan are in 2nd bin of the installments (roughly between 150 and 275 installments)
# <br>
# There is no much outliers with bigger difference and hence no cleanup required

# In[47]:


# Univariate Analysis for int_rate
univ_cont_anlys('int_rate',color_v='b',bin=10)


# ##### Interest Rate Variable Insights
# Maximum number of loan are in 4th bin of the interest rate (roughly between 11.75 and 14.25 interest rate)
# <br>
# There is no much outliers with bigger difference and hence no cleanup required

# In[48]:


# Univariate Analysis for dti
univ_cont_anlys('dti',color_v='b',bin=5)


# ##### dti Variable Insights
# Maximum number of loan are in 3rd bin of the dti (roughly between 12 to 18) and they are moderate risky clients
# <br>
# There is no much outliers with bigger difference and hence no cleanup required

# In[49]:


# # Univariate Analysis for revol_util
# univ_cont_anlys('revol_util',color_v='b',bin=25)


# ##### revol_util Variable Insights
# There is no major difference in the utilization of revolving credit across the bins except 1st bin.
# <br>
# At the same time there are good number of users utilizing the maximum credit which could be risky.
# <br>
# There is no much outliers with bigger difference and hence no cleanup required

# In[50]:


# Univariate Analysis for pub_rec. Print Value counts
print(lend_data['pub_rec'].value_counts())


# In[51]:


univ_cont_anlys('pub_rec',color_v='b',bin=25)


# In[52]:


#Find the % of applicants who has derogatory public records
print(round(lend_data[lend_data['pub_rec']!=0].count().iloc[0]/lend_data['pub_rec'].count()*100,2))


# ##### pub_rec Variable Insights 
# There are around 5.5% of loan applicants who has derogatory public records and they are risky applicants

# In[53]:


#Analyze loan status variable

ax=sns.countplot(lend_data['loan_status'])# count plot loan status

#Display the count in the chart
for p in ax.patches:
    ax.annotate(f'\n{p.get_height()}', (p.get_x()+0.2, p.get_height()), ha='center', va='top', color='white', size=18)
plt.title("Loan Status")
plt.show()


# ##### Loan Status Insights
# 5402 applicats are charged off.

# For below analysis we will consider only Charged off data

# ## Bivariate analysis

# In[54]:


#Draw bar plot for Loan Status Vs Annual Income
sns.barplot(data=lend_data, x="loan_status", y="annual_inc")
plt.title("Loan Status Vs Annual Income")
plt.show()


# 

# #### Loan Status Vs Annual Income Insights
# If annual income is less, charge-off probability is more

# In[55]:


#Draw bar plot for Loan Status Vs Loan Amount
sns.barplot(data=lend_data, x="loan_status", y="loan_amnt")
plt.title("Loan Status Vs Loan Amount")
plt.show()


# #### Loan Status Vs Loan Amount Insights
# If Loan amount is more than 10K, pobability of Charged-off is more

# In[56]:


#Create new data frame only with charged off accounts
lend_data_co=lend_data[lend_data.loan_status == 'Charged Off']


# In[57]:


print(lend_data_co.shape)


# Extract subgrades numbers from the sub_grade variables and plot graph

# In[58]:


lend_data_co["sub_grade_number"] = pd.to_numeric(lend_data.sub_grade.apply(lambda x : x[-1]))


# In[59]:


print("Grade Value Counts")
print(lend_data_co.grade.value_counts())
print("\n Sub-Grade Value Counts")
print(lend_data_co.sub_grade_number.value_counts())


# In[60]:


#Plot the count plot with grades and subgrades in hue for charged account data
fig, ax = plt.subplots(figsize=(12,7))
sns.set_palette('colorblind')
sns.countplot(x = 'grade', order = ['A', 'B', 'C', 'D', 'E', 'F', 'G'] , hue = 'sub_grade_number',data = lend_data_co)
plt.title("Grade & Sub-Grade")
plt.show()


# ##### Grade and Sub-Grade Insights
# Highest number of defaulters are in Grade B irrespective of sub-grades
# <br>
# Highest number of defaulters are sub-Grade B-5

# In[61]:


#Analyze home_ownership variable. find value counts
print(lend_data_co["home_ownership"].value_counts())


# In[62]:


#Plot the pie chart for home ownership and display % of loan applicants spreaded in each category
lend_data_co['home_ownership'].value_counts().plot.pie(autopct='%1.2f%%')
plt.title("Home Ownership Distribution")
plt.show()


# ##### Home Ownership Insights
# Rented home clients are more risker than all.
# <br>
# Mortgage ownership clients are also almost equally risky to Rented home clients. 
# <br>
# So, Rent and Mortgage home ownership clients are more risky in compare to other categories

# In[63]:


#Count plot on purpose variable
sns.countplot(y = 'purpose',data = lend_data_co)
plt.title("Purpose Distribution")
plt.show()


# ##### Purpose of Loan Insights
# Clients who bought loans for Debt-Consolidation purpose are more defaulters

# We will create bins (segment) on requried numeric variables and use as categorical variable for analysis

# In[64]:


print(lend_data_co["loan_amnt"].describe())


# In[65]:


#Create 5 bins with below mentioned range on loan_amnt
lend_data_co['loan_amnt_grp'] = pd.cut(lend_data_co['loan_amnt'], bins=[0,7000,14000,21000,28000,35000],precision =0,labels=['0-7K','7K-14K','14K-21K','21K-28K','28K-35K'])


# In[66]:


#Plot count graph on loan segment
sns.countplot(lend_data_co['loan_amnt_grp'])
plt.title("Loan Amount Range")
plt.show()


# ##### Loan Amount Insights for Charged Off Accounts
# The loan amount range between 0 and 7000 has highest number of defaulters.
# <br>
# The loan amount range between 7000 and 14000 also has similar number of defaulters.

# In[67]:


print(lend_data_co["int_rate"].describe())


# In[68]:


#Create 4 bins with below mentioned range on int_rate and plot count graph on int_rate segments
lend_data_co['int_rate_grp'] = pd.cut(lend_data_co['int_rate'], bins=[5,10,15,20,25],precision =0,labels=['5%-10%','10%-15%','15%-20%','20%-25%'])
sns.countplot(lend_data_co['int_rate_grp'])
plt.title("Interest Rate Range")
plt.show()


# ##### Interest Rate Insights for Charged Off Accounts
# The interest rate range between 10% and 15% has highest number of defaulters.

# In[69]:


print(lend_data_co["annual_inc"].describe())


# In[70]:


#Create 5 bins with below mentioned range on annual_inc and plot count graph on annual_inc segments
lend_data_co['annual_inc_grp'] = pd.cut(lend_data_co['annual_inc'], bins=[4000,31000,58000,85000,112000,141000],precision =0,labels =['4k-31k','31k-58k','58k-85k','85k-112k','112k-141k'])
sns.countplot(lend_data_co['annual_inc_grp'])
plt.title("Annual Income Range")
plt.show()


# ##### Annual Income Insights for Charged Off Accounts
# The clients whose annual income is in the range of 31K to 58K are having highest defaulters

# In[71]:


print(lend_data_co["installment"].describe())


# In[72]:


#Create 10 bins with below mentioned range on installment and plot count graph on installment segments
lend_data_co['installment_grp'] = pd.cut(lend_data_co['installment'], bins=[0,131,262,393,524,655,786,917,1048,1179,1310],  precision =0,labels=['0-131','131-262','262-393','393-524','524-655','655-786','786-917','917-1048','1048-1179','1179-1310'])
fig,ax = plt.subplots(figsize = (15,6))
sns.countplot(lend_data_co['installment_grp'])
plt.title("Installments Range")
plt.show()


# ##### Installments Insights for Charged Off Accounts
# The clients whose installments are between 131-262 are having highest defaulters

# In[73]:


print(lend_data_co["funded_amnt_inv"].describe())


# In[74]:


#Create 7 bins with below mentioned range on funded_amnt_inv_grp and plot count graph on funded_amnt_grp segments
lend_data_co['funded_amnt_inv_grp'] = pd.cut(lend_data_co['funded_amnt_inv'], bins=[0,5000,10000,15000,20000,25000,30000,35000],labels=['0-5k','5k-10k','10k-15k','15k-20k','20k-25k','25k-30k','30k-35k'])
fig,ax = plt.subplots(figsize = (15,6))
sns.countplot(lend_data_co['funded_amnt_inv_grp'])
plt.title("Funded Amount Investor Range")
plt.show()


# #### Funded Amount by Investor Insights for Charged Off Accounts
# The clients whose funded amount by investor is in the range of 5K to 10K are having highest defaulters

# In[75]:


print(lend_data_co["dti"].describe())


# In[76]:


#Create 5 bins with below mentioned range on dti_grp and plot count graph on dti_grp segments
lend_data_co['dti_grp'] = pd.cut(lend_data_co['dti'], bins=[0,6,12,18,24,30],precision =0,labels=['0-6','6-12','12-18','18-24','24-30'])
fig,ax = plt.subplots(figsize = (10,6))
sns.countplot(lend_data_co['dti_grp'])
plt.title("DTI Range")
plt.show()


# ##### DTI Insights for Charged Off Accounts
# The clients whose DTI is in the range of 12-18 are having highest defaulters

# In[77]:


#Count plot term
sns.countplot(lend_data_co["term"])
plt.title("Term")
plt.show()


# ##### Term Insights for Charged Off Accounts
# The clients who took loan tenure of 36 months are having highest number of defaulters

# In[78]:


#Count plot on verification_status
sns.countplot(lend_data_co["verification_status"])
plt.title("Verification Status")
plt.show()

sns.countplot(lend_data["verification_status"])
plt.title("Verification Status")
plt.show()


# ##### Verification Status Insights for Charged Off Accounts
# Non-verified clients are having highest possibilities of defaulting
# 

# In[79]:


#Count plot on Derogatory Public Records values on log scale
fig,ax = plt.subplots(figsize = (10,5))
ax.set_yscale('log')
sns.countplot(lend_data_co["pub_rec"])
plt.title("Derogatory Public Records")
plt.show()


# ##### Derogatory Public Records Insights for Charged Off Accounts
# Though with 0 derogatory public records, clients may fall under defaulters

# #### Anlysing on Date Attributes

# In[80]:


#Count plot on Issue Date Monthly Distribution
sns.countplot(lend_data_co["issue_date_month"])
plt.title("Issue Date Monthly Distribution")
plt.show()


# In[81]:


#Count plot on Issue Date Yearly Distribution
sns.countplot(lend_data_co["issue_date_year"])
plt.title("Issue Date Yearly Distribution")
plt.show()


# ##### Issue Date Insights for Charged Off Accounts
# The loans which are issues in the month of Dec are more defaulted.
# <br>
# The loans which are issues in the year 2011 are more defaulted. It might be because of recession. 

# ### Lets analyze relationships of different variables
# Bin the continuous variables in main data frame where both loan status are available.
# 

# In[ ]:





# In[82]:


#Bin annual_inc,int_rate,loan_amnt & dti variables in below given ranges
lend_data['annual_inc_grp'] = pd.cut(lend_data['annual_inc'], bins=[4000,31000,58000,85000,112000,141000],precision =0,labels =['4k-31k','31k-58k','58k-85k','85k-112k','112k-141k'])
lend_data['int_rate_grp'] = pd.cut(lend_data['int_rate'], bins=[5,10,15,20,25],precision =0,labels=['5%-10%','10%-15%','15%-20%','20%-25%'])
lend_data['loan_amnt_grp'] = pd.cut(lend_data['loan_amnt'], bins=[0,7000,14000,21000,28000,35000],precision =0,labels=['0-7K','7K-14K','14K-21K','21K-28K','28K-35K'])
lend_data['dti_grp'] = pd.cut(lend_data['dti'], bins=[0,6,12,18,24,30],precision =0,labels=['0-6','6-12','12-18','18-24','24-30'])


# ### Multivariate analysis

# In[83]:


#Analysis of Loan Amount Vs Term wrt loan status
plt.figure(figsize=(10,7))
sns.barplot(data=lend_data, x='loan_amnt_grp', y='term',hue='loan_status') #Draw bar plot for given variables
plt.title("Laon Amount Vs Term Vs Loan Status")
plt.show()


# In[84]:


#Analysis of Loan Amount Vs Annual Income wrt loan status
plt.figure(figsize=(10,7))
sns.barplot(data=lend_data, x='annual_inc_grp', y='loan_amnt',hue='loan_status')
plt.title("Laon Amount Vs Annual Income Vs Loan Status")
plt.show()


# In[85]:


#Analysis of Annual Income Vs Interest Rate wrt loan status
plt.figure(figsize=(10,7))
sns.barplot(data=lend_data, x='annual_inc_grp', y='int_rate',hue='loan_status')
plt.title("Annual Income Vs Interest Rate Vs Loan Status")
plt.show()


# In[86]:


#Analysis of Loan Amount Vs Interest Rate wrt loan status
plt.figure(figsize=(10,7))
sns.barplot(data=lend_data, x='loan_amnt_grp', y='int_rate',hue='loan_status')
plt.title("Loan Amount Vs Interest Rate Vs Loan Status")
plt.show()


# In[87]:


#Analysis of Annual Income Vs DTI wrt loan status
plt.figure(figsize=(12,7))
sns.barplot(data=lend_data, x='annual_inc_grp', y='dti',hue='loan_status')
plt.title("Annual Income Vs DTI Vs Loan Status")
plt.show()


# In[88]:


#Analysis of Loan Amount Vs Purpose wrt loan status
plt.figure(figsize=(10,10))
sns.barplot(data =lend_data,x='loan_amnt', y='purpose', hue ='loan_status',palette="crest")

plt.title("Loan Amount Vs Purpose Vs Loan Status")
plt.show()


# In[89]:


#Draw bar plot for analysis of Annual Income Vs Purpose wrt loan status
plt.figure(figsize=(10,10))
sns.barplot(data =lend_data,x='annual_inc', y='purpose', hue ='loan_status',palette="crest")
plt.title("Annual Income Vs Purpose Vs Loan Status")
plt.show()


# In[90]:


#Draw bar plot for analysis of Loan Amount Vs Verification Status wrt loan status
plt.figure(figsize=(10,10))
sns.barplot(data =lend_data,x='verification_status', y='loan_amnt',  hue ='loan_status',palette="mako")
plt.title("Loan Amount Vs Verification Status Vs Loan Status")
plt.show()


# In[91]:


#Draw bar plot for analysis of Annual Income Vs Verification Status wrt loan status
plt.figure(figsize=(10,10))
sns.barplot(data =lend_data,x='verification_status',y='annual_inc',  hue ='loan_status',palette="mako")
plt.title("Annual Income Vs Verification Status Vs Loan Status")
plt.show()


# In[92]:


#Draw bar plot for analysis of Loan Amount Vs Grade wrt loan status
plt.figure(figsize=(10,10))
sns.barplot(data =lend_data,x='grade', y='loan_amnt', hue ='loan_status',palette="magma")
plt.title("Loan Amount Vs Grade Vs Loan Status")
plt.show()


# In[93]:


#replacing 'NONE' with 'OTHER' in home ownership as there are only few records of None
lend_data['home_ownership']=lend_data['home_ownership'].str.replace('NONE','OTHER')


# In[94]:


#Draw bar plot for analysis of Loan Amount Vs Home Ownership wrt loan status
plt.figure(figsize=(10,10))
sns.barplot(data =lend_data,x='home_ownership', y='loan_amnt', hue ='loan_status',palette="magma")
plt.title("Loan Amount Vs Home Ownership Vs Loan Status")
plt.show()


# In[95]:


#Draw bar plot for analysis of Annual Income Vs Home Ownership wrt loan status
plt.figure(figsize=(10,10))
sns.barplot(data =lend_data,x='home_ownership', y='annual_inc', hue ='loan_status',palette="magma")
plt.title("Annual Income Vs Home Ownership Vs Loan Status")
plt.show()


# In[96]:


#Draw bar plot for analysis of Loan Amount Vs Employment Length wrt loan status
plt.figure(figsize=(20,20))
sns.barplot(data =lend_data,x='emp_length',y='loan_amnt',  hue ='loan_status',palette="mako")
plt.title("Loan Amount Vs Employment Length Vs Loan Status")
plt.show()


# In[97]:


#Draw bar plot for analysis of Annual Income Vs Employment Length wrt loan status
plt.figure(figsize=(10,10))
sns.barplot(data =lend_data,x='emp_length', y='annual_inc',  hue ='loan_status',palette="mako")
plt.title("Annual Income Vs Employment Length Vs Loan Status")
plt.show()


# In[98]:


#Draw bar plot for analysis of Interest Rate and Grade wrt loan status
plt.figure(figsize=(10,10))
sns.barplot(data =lend_data,x='grade', y='int_rate',  hue ='loan_status',palette="mako")
plt.title("Interest Rate Vs Grade Vs Loan Status")
plt.show()


# In[99]:


#Draw bar plot for analysis of Loan Amount and state wrt loan status
plt.figure(figsize=(20,30))
sns.barplot(data =lend_data,y='addr_state', x='loan_amnt', hue ='loan_status',palette="mako")
plt.title("Loan Amount Vs State Vs Loan Status")
plt.show()


# ##### Consolidated Insights of Variable Relationships wrt Loan Status
# <br>
# With this analysis, we can conclude that probability of defaulters could be because of below reasons.
# <br>
# 1.	Client has loan amount in the range of 28000 to 35000 with term more than 50 months
# <br>
# 2.	Client has annual income above 112k-141K with loan amount of more than 15K
# <br>
# 3.	Client with rate of interest more than 11% irrespective of annual income
# <br>
# 4.	Client with loan amount in the range of 28000 to 35000 with interest rate more than 15% 
# <br>
# 5.	Client with annual income in the range of 31K to 58K with dti more than 14
# <br>
# 6.	Clients requesting small business loans with loan amount above 14000
# <br>
# 7.	Clients requesting home improvement loan with annual income of above 65000
# <br>
# 8.	Verified loan account client requesting loan amount more than 15000 or annual income of above 60000
# <br>
# 9.	Client with grade F & G and has loan amount more than 15000. This could be because of high interest rate and loan amount disposed
# <br>
# 10.	Client whose home ownership is Mortgage or Others and has loan amount more than 12000
# <br>
# 11.	Client whose home ownership is Mortgage and has annual income above 60000
# <br>
# 12.	More the employment experience, loan amount keeps increasing and defaulters are also high in all experience level. Ratio between defaulters and fully paid is not much. If lender provides loan amount around 1K-2K less across the experience level, defaulters will be less.
# <br>
# 13.	Clients in address state “WY” with loan amount more than 15000
# 

# In[100]:


#Convert the loan status to numeric variables to check the correlation 0 for Charged Off accounts and 1 for fully paid
lend_data['loan_status_flag'] = lend_data['loan_status'].map(lambda x: 0 if x == 'Charged Off' else 1).astype(int)


# In[101]:


import matplotlib.pyplot as plt
import seaborn as sns

# Select only numeric columns
numeric_data = lend_data.select_dtypes(include=[np.number])

# Plot heatmap on all numerical variables for correlation analysis
plt.figure(figsize=(15, 15))
sns.heatmap(numeric_data.corr(), cmap="Reds", annot=True)
plt.title("Heat Map")
plt.show()

# Display correlation matrix
print(numeric_data.corr())


# ##### Heat Map - Correlation Insights
# Loan Amount, Funded value investor, annual income, Installemts, interest rate, revolving utilization, and term are correlated loan status determination

# In[102]:


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))

