# -*- coding: utf-8 -*-

# to handle datasets
import pandas as pd
import numpy as np
# for plotting
import matplotlib.pyplot as plt
#% matplotlib inline

# to display all the columns of the dataframe in the notebook
pd.pandas.set_option('display.max_columns', None)

# load dataset
data = pd.read_csv('candy-data.csv',encoding='utf-8')

# rows and columns of the data
print(data.shape)

# visualise the dataset
data.head()

#get all the missing values
data.isnull().sum()

#segregating Dependent and Independent variables
X = data.iloc[:, 1:12].values
y = data.iloc[:, 12].values

#-----------------------------------------------------------------------------
# Splitting the dataset into the Training set and Test set
from sklearn import datasets, linear_model
from sklearn.cross_validation import train_test_split
from matplotlib import pyplot as plt
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Predicting the results
y_pred=regressor.predict(X_test)



#Building the optimal model using backward elimination with signifcant level as 0.05(5%)
import statsmodels.formula.api as sm
X=np.append(arr=np.ones((85,1)).astype(int),values=X, axis=1)
#Regression with all the IDV--11IDV
X_opt=X[:,[0,1,2,3,4,5,6,7,8,9,10,11]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Removing x8 as it has the maximum p-value--10IDV
X_opt=X[:,[0,1,2,3,4,5,6,7,9,10,11]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Removing x5 as it has the maximum p-value--9IDV
X_opt=X[:,[0,1,2,3,4,6,7,9,10,11]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Removing x7 as it has the maximum p-value--8IDV
X_opt=X[:,[0,1,2,3,4,6,7,10,11]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Removing x3 as it has the maximum p-value--7IDV
X_opt=X[:,[0,1,2,4,6,7,10,11]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()

#Removing x7 as it has the maximum p-value--6IDV
X_opt=X[:,[0,1,2,4,6,7,10]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#Adjusted R-square value decrased from 0.493 to 0.492 when shifting from 7 IDV to 6 IDV so model with the below 7 IDV looks to be a good bet.


#The model performs best with 7 IDV
X_opt=X[:,[0,1,2,4,6,7,10,11]]
regressor_OLS=sm.OLS(endog=y,exog=X_opt).fit()
regressor_OLS.summary()
#x1=(index=1)=chocolate
#x2=(index=2)=fruity
#x3=(index=4)=peanutyalmondy
#x4=(index=6)=crispedricewafer
#x5=(index=7)=hard
#x6=(index=10)=sugarpercentage
#x7=(index=11)=pricepercentage

# From the cofficient value it can be derived that price and hard has negative corelation with win percentage. So these substances need to be avioded.
# chocolate, fruity,peanutyalomdy, crispedricewater has positive relation with win percentage so a candy with this behaviour can create more value for customer.
