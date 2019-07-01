#!/usr/bin/env python
# coding: utf-8

# In[197]:


import warnings
warnings.filterwarnings('ignore')


# In[198]:



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


# In[199]:


boston_data = load_boston()


# In[200]:


boston_data.keys()


# In[201]:


boston_data.DESCR


# In[ ]:





# In[202]:


boston_data.feature_names


# In[203]:


import pandas
import numpy
data = pandas.DataFrame(boston_data.data,columns=boston_data.feature_names)


# In[204]:


data.head()


# In[205]:


data.isnull().sum()


# In[206]:


data


# In[207]:


Xdata = data[data.columns[0:12]]
Ydata = data[data.columns[12:13]]
Xdata


# In[208]:


import seaborn


# In[209]:


seaborn.heatmap(data.corr())


# In[210]:


absMat = Xdata.corr().abs()
Utri = absMat.where(numpy.triu(numpy.ones(absMat.shape),k=1).astype(numpy.bool))

corrFea = [column for column in Utri.columns if any(Utri[column]>0.75)]

print(corrFea)


# In[211]:


Xdata =Xdata.drop(corrFea,axis=1)


# In[212]:


Xdata


# In[213]:


Xtrain,Xtest,Ytrain,Ytest = train_test_split(Xdata,Ydata,test_size=0.2)
#liRegress= LinearRegression()
#liRegress.fit(pandas.DataFrame(Xtrain),pandas.DataFrame(Ytrain))

#Ypred = liRegress.predict(Xtest)
LiRegression = LinearRegression()

LiRegression.fit(pandas.DataFrame(Xtrain),pandas.DataFrame(Ytrain))
Ypred = LiRegression.predict(pandas.DataFrame(Xtest))


# In[214]:


print("Error       : %.lf" %mean_squared_error(Ytest,Ypred))
print("R2 Score :  %.2lf" %r2_score(Ytest,Ypred))
print("Mean absolute Error : %.2f"%mean_absolute_error(Ytest,Ypred))


# In[ ]:





# In[ ]:




