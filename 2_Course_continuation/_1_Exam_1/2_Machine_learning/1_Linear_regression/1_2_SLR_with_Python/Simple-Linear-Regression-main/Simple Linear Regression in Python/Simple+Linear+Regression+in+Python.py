# Supress Warnings

import warnings
warnings.filterwarnings('ignore')

# Import the numpy and pandas package

import numpy as np
import pandas as pd

# Read the given CSV file, and view some sample records

advertising = pd.read_csv("https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_1_Exam_1/2_Machine_learning/1_Linear_regression/1_2_SLR_with_Python/Simple-Linear-Regression-main/Simple%20Linear%20Regression%20in%20Python/advertising.csv")
advertising.head()

advertising.shape

advertising.info()

advertising.describe()

import matplotlib.pyplot as plt 
import seaborn as sns

sns.pairplot(advertising, x_vars=['TV', 'Newspaper', 'Radio'], y_vars='Sales',size=4, aspect=1, kind='scatter')
plt.show()

sns.heatmap(advertising.corr(), cmap="YlGnBu", annot = True)
plt.show()

X = advertising['TV']
y = advertising['Sales']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)

# Let's now take a look at the train dataset

X_train.head()

y_train.head()

get_ipython().system('pip install statsmodels')

import statsmodels.api as sm

# Add a constant to get an intercept
X_train_sm = sm.add_constant(X_train)

# Fit the resgression line using 'OLS'
lr = sm.OLS(y_train, X_train_sm).fit()

# Print the parameters, i.e. the intercept and the slope of the regression line fitted
lr.params

# Performing a summary operation lists out all the different parameters of the regression line fitted
print(lr.summary())

plt.scatter(X_train, y_train)
plt.plot(X_train, 6.948 + 0.054*X_train, 'r')
plt.show()

y_train_pred = lr.predict(X_train_sm)
res = (y_train - y_train_pred)

fig = plt.figure()
sns.distplot(res, bins = 15)
fig.suptitle('Error Terms', fontsize = 15)                  # Plot heading 
plt.xlabel('y_train - y_train_pred', fontsize = 15)         # X-label
plt.show()

plt.scatter(X_train,res)
plt.show()

# Add a constant to X_test
X_test_sm = sm.add_constant(X_test)

# Predict the y values corresponding to X_test_sm
y_pred = lr.predict(X_test_sm)

y_pred.head()

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

#Returns the mean squared error; we'll take a square root
np.sqrt(mean_squared_error(y_test, y_pred))

r_squared = r2_score(y_test, y_pred)
r_squared

plt.scatter(X_test, y_test)
plt.plot(X_test, 6.948 + 0.054 * X_test, 'r')
plt.show()

from sklearn.model_selection import train_test_split
X_train_lm, X_test_lm, y_train_lm, y_test_lm = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)

X_train_lm.shape

# Assuming X_train_lm and X_test_lm are pandas Series
X_train_lm = X_train_lm.values.reshape(-1, 1)
X_test_lm = X_test_lm.values.reshape(-1, 1)

print(X_train_lm.shape)
print(y_train_lm.shape)
print(X_test_lm.shape)
print(y_test_lm.shape)

from sklearn.linear_model import LinearRegression

# Representing LinearRegression as lr(Creating LinearRegression Object)
lm = LinearRegression()

# Fit the model using lr.fit()
lm.fit(X_train_lm, y_train_lm)

print(lm.intercept_)
print(lm.coef_)

corrs = np.corrcoef(X_train, y_train)
print(corrs)

corrs[0,1] ** 2

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.7, test_size = 0.3, random_state = 100)

from sklearn.preprocessing import StandardScaler, MinMaxScaler

# One aspect that you need to take care of is that the 'fit_transform' can be performed on 2D arrays only. So you need to
# reshape your 'X_train_scaled' and 'y_trained_scaled' data in order to perform the standardisation.
X_train_scaled = X_train.values.reshape(-1,1)
y_train_scaled = y_train.values.reshape(-1,1)

X_train_scaled.shape

# Create a scaler object using StandardScaler()
scaler = StandardScaler()
#'Fit' and transform the train set; and transform using the fit on the test set later
X_train_scaled = scaler.fit_transform(X_train_scaled)
y_train_scaled = scaler.fit_transform(y_train_scaled)

print("mean and sd for X_train_scaled:", np.mean(X_train_scaled), np.std(X_train_scaled))
print("mean and sd for y_train_scaled:", np.mean(y_train_scaled), np.std(y_train_scaled))

# Let's fit the regression line following exactly the same steps as done before
X_train_scaled = sm.add_constant(X_train_scaled)

lr_scaled = sm.OLS(y_train_scaled, X_train_scaled).fit()

# Check the parameters
lr_scaled.params

print(lr_scaled.summary())

# import packages
# How to Calculate Residual Sum of Squares in Python
# https://www.geeksforgeeks.org/how-to-calculate-residual-sum-of-squares-in-python/

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression


# reading csv file as pandas dataframe
data = pd.read_csv('https://raw.githubusercontent.com/aqwertyuiop48/upgrad_programming/refs/heads/main/2_Course_continuation/_1_Exam_1/2_Machine_learning/1_Linear_regression/1_2_SLR_with_Python/Simple-Linear-Regression-main/Simple%20Linear%20Regression%20in%20Python/headbrain2.csv')

# independent variable
X = data[['Head Size(cm^3)']]

# output variable (dependent)
y = data['Brain Weight(grams)']

# using the linear regression model
model = LinearRegression()

# fitting the data
model.fit(X, y)

# predicting values
y_pred = model.predict(X)
df = pd.DataFrame({'Actual': y, 'Predicted':
y_pred})

print(' residual sum of squares is : '+ str(np.sum(np.square(df['Predicted'] - df['Actual']))))


import datetime, pytz; 
print("Current Time in IST:", datetime.datetime.now(pytz.utc).astimezone(pytz.timezone('Asia/Kolkata')).strftime('%Y-%m-%d %H:%M:%S'))



