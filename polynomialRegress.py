#!/usr/bin/env python
# coding: utf-8

# In[18]:


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


# In[5]:


data = pandas.read_csv('.\Desktop\qaz\data\overfitting_data.csv')
data.head()


# In[10]:


plt.scatter(data['X'],data['Y'])


# In[14]:


Xdata= data['X']
Ydata= data['Y']

Xtrain, Xtest,Ytrain,Ytest = train_test_split(Xdata,Ydata,test_size=0.3)


# In[55]:


def LiRegDegree(Xtrain,Ytrain,Xtest,Ytest,degree):
    model =make_pipeline(PolynomialFeatures(degree),LinearRegression())
    model.fit(pandas.DataFrame(Xtrain),pandas.DataFrame(Ytrain))
    
    Xtest=pandas.DataFrame(Xtest,columns=['X'])
    Xtest=Xtest.sort_values(by=['X'])
    Ytest=pandas.DataFrame(Ytest)
    Ytest=Ytest.ix[Xtest.index]
    
    Ypred=model.predict(Xtest)
    smooth_feature=numpy.linspace(numpy.min(Xtest['X'].tolist()),numpy.max(Xtest['X'].tolist()),1000)
    smooth_points=spline(Xtest['X'].tolist(),Ypred,smooth_feature)
    
    plt.plot(smooth_feature,smooth_points,'-g')
    plt.scatter(Xtrain,Ytrain)
    plt.title('Mean absolute error : %.4f' %mean_absolute_error(Ytest,Ypred))
    
    


# In[60]:


LiRegDegree(Xtrain,Ytrain,Xtest,Ytest,4)


# In[62]:


#RIDGE REGRSSION FROM HERE


# In[85]:


boston_data = load_boston()
Bdata= pandas.DataFrame(boston_data.data,columns=boston_data.feature_names)
BXdata = Bdata[Bdata.columns[0:12]]
BYdata = Bdata[Bdata.columns[12:13]]
BXtrain,BXtest,BYtrain,BYtest = train_test_split(pandas.DataFrame(BXdata),pandas.DataFrame(BYdata),test_size=0.2)


# In[91]:


ridgeReg = Ridge()
ridgeReg.fit(BXtrain,BYtrain)
Bpredm = ridgeReg.predict(BXtest)

print('mean absolute error : %.2f' %mean_absolute_error(BYtest,Bpredm))

