#!/usr/bin/env python
# coding: utf-8

# # Title: Machine Learning for fuel consumption prediction  

# # Content of the document
#  1. Problem Statement 
#  2. Data Collection
#  3. Data Exploration
#  4. Data Preprocessing   
#  5. Model selection and hyperparameter tuning
#  5. Model Assessement
#  6. Feature Importance Analysis
#  7. Conclusion
#  8. References 
# 
# 

# # 1. Problem Statement
# 
# The objective of this project is <b>to predict the fuel efficiency of vehicles (MPG)</b> based on the other information about the vehicles. My company provided me with historical continuous data on MPG based on the fuel efficiency of each vehicle from the 70s to the 80s.
# 
# In order to accomplish this, I need to <b>create an end-to-end supervised machine learning pipeline </b>. Once the pipeline is designed and implemented, it will be submitted to the company's lead data scientist for prediction purposes.
# 
# 

# Here are the steps I will take to build my pipeline: 
#     
#      1. Data Collection: I will use the  Auto MPG dataset obtained from the UCI ML Repository.
#      2. Data Exploration: This will be done to identify the most important features and combine them in new ways.
#      3. Data Preprocessing: Lay out a pipeline of tasks for transforming data for use in my machine learning model.
#      4. Model selection & Hyperparameter Tuning : Cross-validate a few models and fine-tune hyperparameters for 
#         models that showed promising predictions.
#      5. Model Assessment: Determine the performance of the final trained model.
#      6. A feature importance analysis
#      7. Conclusion & recommendations 
#     

# # 2. Data Collection
# In this step I will: 
#     
#   - Identify data sources
#   - Split the data into training and test sets
# 
# 
# Before starting, as a first step, I will call some libraries I need in order to build my model.
# 

# In[1]:


import warnings
warnings.filterwarnings('ignore')
#install the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import StratifiedShuffleSplit
# import linear regression
from sklearn.linear_model import LinearRegression
# Import mean squared error
from sklearn.metrics import mean_squared_error
# Import Grid search CV
from sklearn.model_selection import GridSearchCV
# Import the SVR
from sklearn.svm import SVR
#import Random Forest Regressor
from sklearn.ensemble import RandomForestRegressor


# Source of the data: (UCI Machine Learning Repository: Auto MPG Data Set, 2022)

# In[2]:


# Load the data from UCI ML Repository

get_ipython().system('wget "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data"')


# In[3]:


# Using Pandas to read data from a file

attributes = ['mpg','cylinders','displacement','horsepower','weight','Speed', 'year model', 'origin']

initial_data = pd.read_csv('./auto-mpg.data', names=attributes, na_values = "?", comment = '\t', sep= " ",
                           skipinitialspace=True)


# In[4]:


# Create a copy of the original data
my_data = initial_data.copy()

# Examine my data structure and return the top 5 rows of the data frame.
my_data.head(5)


# In[5]:


#Split my data into training and test sets
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for tr_ind, test_ind in split.split(my_data, my_data["cylinders"]):
    tr_set = my_data.loc[tr_ind]
    test_set = my_data.loc[test_ind]


# In[6]:


# Segregating Target and Feature variables
data_set = tr_set.drop("mpg", axis=1)
target = tr_set["mpg"].copy()


# In[7]:


data_set.info()


# # 3. Data Exploration
#  

# ### Check for Data type of columns

# In[8]:


# Check the info of my data
data_set.info()


# 4 values are missing from the variable "horsepower". As far as the formatting is concerned, nothing needs to be done.

#  ### Check for null values

# In[9]:


# Looking for all the null values
data_set.isnull().sum()


# It has been mentioned earlier that only <b> "horsepower" has four missing values </b>. 

# In[10]:


### Check summary statistics
data_set.describe()


# ### Look for the category distribution in categorical columns

# Now I want to see the distribution to know the % of how many rows belong to a particulare class of value.
# To do that I will first count the number of rows for each class of value then I will devide it by the total 
# number of rows.
# In my case I will do that for both "origin" & "cylinders"

# In[11]:


## Origin distribution
data_set['origin'].value_counts()/ len(data_set)


# According to the results, more than 62% of the origin "1", 29% from "2" and 18% from "3".

# In[12]:


## Cylinders distribution
data_set["cylinders"].value_counts() / len(data_set)


# According to the results, more than 50% of the engines are 4 cylinders, 25% are 8 cylinders, 21% are 6 cylinders, and the remaining are 3 cylinders and 5 cylinders.
# 
# My consideration of both distributions leads me to keep in mind that <b>while testing that most of the vehicles belong to 4 cylinders & are mostly from origin 1</b> 

# ### Checking correlation between different attributes 

# To do that I will use the function Corr of Pandas

# In[13]:


data_set.corr().style.background_gradient(cmap="GnBu")


# This helps to understand witch are the most important features to look at when building my machine learning

# # 4. Data Preprocessing

# Choosing the best imputation technique (mean, median or mode)is key to getting the best value from missing values. 
# Using this value, missing values can be replaced appropriately by finding out which measures
# the central tendency best. <b>(python, 2022)</b>
# 
# A distribution plot or a box plot is extremely useful for determining which technique to use. For that we use 
# the function sns.boxplot as follow

# In[14]:


sns.boxplot(x=data_set['horsepower'], color='yellow')


# Considering there are only a few outliers, I opted to <b> impute null values based on the median</b>

# In[15]:


# calculate the median
my_median = data_set['horsepower'].median()


# ### Impute null values of "horsepower"
# 

# In[16]:


#impute my null values with median
data_set['horsepower'] = data_set['horsepower'].fillna(my_median)


# In[17]:


# Check my new values
data_set.info()


# # 4. Selecting and Training Models
# In this section I will train the 3 following models, train them and compare between them:
#  
#     - Linear Regression
#     - Random Forest
#     - Support Vector Machine regressor
#     

# ## Linear Regression

# In[18]:


linear_reg = LinearRegression()
linear_reg.fit(data_set, target)


# In[19]:


# Testing the predictions 
sample_mydata = data_set.iloc[:10]
sample_target = target.iloc[:10]

print("Prediction of samples: ", linear_reg.predict(sample_mydata))


# In[20]:


print("Actual Labels of samples: ", list(sample_target))


# ### Calculate the Mean Squared Error

# In[21]:


mpg_pred = linear_reg.predict(data_set)
mse_linear = mean_squared_error(target, mpg_pred)
rmse_linear = np.sqrt(mse_linear)
print('The mean squared error is for linear regression model is:')
rmse_linear


# ### Cross validation for linear regression model

# When Scikit-Learn performs a K-fold cross-validation, the training set is randomly split into K subsets called folds, and then the model is trained and evaluated K times, with each fold being evaluated at a different time, and each fold being trained on the following time.
# 
# The result is an array containing the scores for all K evaluations:

# In[22]:


from sklearn.model_selection import cross_val_score

#Pass linear regression model & prepare the data labels scoring method and then 10 quick k-fold cross validation
scor = cross_val_score(linear_reg, data_set, target, scoring="neg_mean_squared_error", cv = 10)
linear_reg_scor_rmse = np.sqrt(-scor)
print('The mean square error values of the 10 quick K-fold cross validations:')
linear_reg_scor_rmse


# In[23]:


# Find out the average
print('The average mean square error for Linear regression model: ')
linear_reg_scor_rmse.mean()


# ## Random Forest model

# In[24]:


# Utilize the fit method to initiate training
regress_forst = RandomForestRegressor()
regress_forst.fit(data_set, target)

#Provide the cross value score 
forest_reg_cv_scor= cross_val_score(regress_forst,
                                         data_set,
                                         target,
                                         scoring='neg_mean_squared_error',
                                         cv = 15)

# For all 10 values I have, calculate the square root of my negative values
forest_reg_rmse_scor = np.sqrt(-forest_reg_cv_scor)


# In[25]:


# Calculate the average 
print('The average mean square error for Random Forest Regressor : ')
forest_reg_rmse_scor.mean()


# <b> Random Forest performed better </b>than the linear regression model 

# ## Support Vector Machine Regressor

# In[26]:


# I have selected linear to map a lower dimensional data into a higher dimensional data
regr_svm = SVR(kernel='linear')
# fit the data with fit function
regr_svm.fit(data_set, target)
#cross validation
regr_svm_cv_scor = cross_val_score(regr_svm, data_set, target,
                                scoring='neg_mean_squared_error',
                                cv = 15)

rmse_scor_svm = np.sqrt(-regr_svm_cv_scor)


# In[27]:


# Calculate the average 
print('The average mean square error for SVMR : ')
rmse_scor_svm.mean()


# So far we see Random Forest turns out to be the best model out of the 3. Now I will perform Hyperparameter tuning to find out which set of parameters of the random forest model works the best. So if we can improve the performane of random forest model from what we already have. 

# ## GridSearchCV for hyperparameter tuning

# The hyperparameters of the random forest regressor must be fine-tuned here. In order to do so, I selected the grid search of the cyclic learns model selection module. 

# In[28]:


# define the parameter grid 
prm_grid_ = [
    {'n_estimators': [2, 10, 15], 'max_features': [2, 4, 6,8]},
    {'bootstrap': [False], 'n_estimators': [4, 8], 'max_features': [2, 4, 5]},
  ]

frst_regres = RandomForestRegressor()

search_grid = GridSearchCV(frst_regres, prm_grid_,
                           scoring='neg_mean_squared_error',
                           return_train_score=True,
                           cv=10,
                          )
# Fit the data 
search_grid.fit(data_set, target)


# In[29]:


print("The best parameters we could have for Random Forest are:")
search_grid.best_params_


# Now we want to see which parameters had returned what scores 

# In[30]:


# Keeping track of all our scores
scor_cv = search_grid.cv_results_

# Print all the parameters along with their scores
for scor_mean, prms in zip(scor_cv['mean_test_score'], scor_cv["params"]):
    print(np.sqrt(-scor_mean), prms)


# I still have <b>my best model, the Random Forest Regressor<b/>, with a square error of 2.51.

# # 5. Model Assessement

# In order to assess the model using the data I kept for testing. First ,  I must prepare it and ensure that there are no null values.

# In[31]:


test_set.info()


# Using the same approach I applied to the preprocessing data step, I will fill in the two missing values for the attribute horsepower.

# In[32]:


# calculate the median
test_median = test_set['horsepower'].median()
#impute my null values with median
test_set['horsepower'] = test_set['horsepower'].fillna(test_median)
# Check my new values
test_set.info()


# The time has come to test my model, and I have chosen the Random Forest Regressor as my model. 

# In[33]:


# capture my best model in selected model variable 
selected_model = search_grid.best_estimator_


# drop the mpg from our test data
data_test = test_set.drop("mpg", axis=1)

#segregate my mpg from my testing data 
target_test = test_set["mpg"].copy()


# In[34]:


#Predict the result
selected_model_pr = selected_model.predict(data_test)

#calculate squared error
mse_last = mean_squared_error(target_test, selected_model_pr)
rmse_last=np.sqrt(mse_last)


# In[35]:


#Print 
rmse_last


# It is encouraging to see that the squared error has decreased from 2.81 to 1.27 compared to the training one. 

# In[36]:


# Testing the predictions using my test data
sample_testdata = data_test.iloc[:5]
sample_testtarget = target_test.iloc[:5]

print("Prediction of samples with the my selected model: ", selected_model.predict(sample_testdata))


# In[37]:


print("Actual Labels of samples: ", list(sample_testtarget))


# Based on my testing data, I consider the model chosen to be good

# # 6. Feature importance Analysis

# In[38]:


# calculate features importance 
feature_import = search_grid.best_estimator_.feature_importances_
feature_import


# We cannot make sense of these numbers if we keep them in this manner without knowing which features they belong to. To do that, I'll combine their names with features' names

# In[39]:


# With the reverse method, the most important feature will appear at the top and so on
print("Features importance:")
sorted(zip(attributes, feature_import), reverse=True)


# The year model appears to be the most important feature based on the results above. It is now time to evaluate our model with test data.

# # 8. Conclusion 
# As a result, the machine created for the company can be an effective solution. It may not be 100% accurate, but it can be improved since the squared error was only 2.81 and on testing data we saw a significant improvement of 1.27. Thus, machine learning needs to be trained better to reduce errors. With this machine, the company can start working right away. 

# # 9.References

# 2022. python. [online] Available at: <https://vitalflux.com/pandas-impute-missing-values-mean-median-mode/
# #:~:text=When%20the%20data%20is%20skewed,be%20done%20with%20numerical%20data> [Accessed 8 July 2022].

# Archive.ics.uci.edu. 2022. UCI Machine Learning Repository: Auto MPG Data Set. [online] Available at: <http://archive.ics.uci.edu/ml/datasets/Auto+MPG> [Accessed 5 July 2022].
