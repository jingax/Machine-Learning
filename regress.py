#!/usr/bin/env python
# coding: utf-8

# In[79]:


import warnings
warnings.filterwarnings('ignore')


# In[80]:


import numpy
import pandas
import sklearn
import seaborn
import matplotlib.pyplot as plt
#%matplot inline
from pylab import rcParams
rcParams['figure.figsize'] = 10, 8

from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression, Lasso,Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import KFold
from sklearn.datasets import load_boston
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
from scipy.interpolate import spline
from sklearn.neighbors import KNeighborsRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import train_test_split


# In[81]:


data = pandas.read_csv('.\Desktop\dataSet.csv')


# In[82]:


data.head()


# In[83]:


plt.scatter(data['yr_built'],data['price'])
plt.xlabel('Area')
plt.ylabel('Price')
plt.title('Price-Area Data')


# In[84]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(data['sqft_living'],data['price'],test_size=0.20)


# In[85]:


LiRegression = LinearRegression()
#Xtrain = Xtrain.values
#Ytrain = Ytrain.values
#Xtest = Xtest.values
LiRegression.fit(pandas.DataFrame(Xtrain),pandas.DataFrame(Ytrain))

Ypred = LiRegression.predict(pandas.DataFrame(Xtest))
Ytrain.describe()


# In[86]:


print("Error       : %.lf" %mean_squared_error(Ytest,Ypred))


# In[87]:


print("R2 Score :  %.2lf" %r2_score(Ytest,Ypred))


# In[88]:


plt.scatter(Xtest,Ytest,color='blue')
plt.plot(Xtest,Ypred,color='red')

