#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


# In[125]:


class LinearRegressionModel:
    def __init__(self):
        pass
    
    def fit(self, X, y):
        self.X = X;
        self.y = y;
        
        # Matrix Form of linear regression
        self.X = np.vstack((np.ones(X.shape[0]), X)).T
        self.y = self.y.T
        
        
        X_T_X = self.X.T @ self.X
        X_T_X_inv = np.linalg.inv(X_T_X)
        X_T_Y = self.X.T @ self.y
        
        self.coefficients = X_T_X_inv @ X_T_Y
        return self.coefficients
    
    def predict(self, x):
        return np.dot(x, self.coefficients[1:]) + self.coefficients[0];
    
    def plot(self, points = 10):
        min_val = min(self.X[:, 1])
        max_val = max(self.X[:, 1])
        X = np.arange(min_val, max_val, points/(max_val - min_val))
        plt.plot(X, [self.predict(x) for x in X]);
        plt.scatter(self.X[:, 1].T, self.y.T)
        plt.show()
        
        


# In[131]:


df = pd.read_csv('./data.csv')


# In[132]:


data = df.to_numpy()


# In[133]:


X = data[:, 0]
y = data[:, 1]


# In[134]:


X.shape


# In[135]:


model = LinearRegressionModel()
regression_coefficients = model.fit(X, y)
regression_coefficients


# In[136]:


model.coefficients


# In[137]:


X,y


# In[138]:


model.predict(11)


# In[139]:


model.plot()


# In[ ]:




