#!/usr/bin/env python
# coding: utf-8

# In[107]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import math
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# In[108]:


class Distance:
    
    @staticmethod
    def Minkowski(x, y, p = 2):
        return np.sum((x-y)**p)**(1/p)
    
    @staticmethod
    def Euclidean(x, y):
        return math.sqrt(np.sum((x-y)**2))
    
    @staticmethod
    def Manhattan(x, y):
        return np.sum(np.abs(x-y))
    
    @staticmethod
    def Chebyshev(x, y):
        return max(np.abs(x-y))


# In[109]:


class KNN:
    def __init__(self):
        pass
    
    def fit(self, X, y, k = None):
        self.X = X
        self.y = y
        self.k = self.X.shape[0]//4 if k is None else k
        
    def predict(self, x_hat, distance_fn = Distance.Euclidean):
        nearest = np.argsort([distance_fn(x_hat, x) for x in self.X])[:self.k]
        classes, counts = np.unique(self.y[nearest], return_counts = True)
        
        return classes[np.argmax(counts)]
            


# In[140]:


iris = load_iris()

X = iris['data']
y = iris['target']


# In[141]:


x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, shuffle = True, random_state = 46)


# In[142]:


np.unique(y_train, return_counts = True)


# In[143]:


model = KNN()


# In[144]:


model.fit(x_train, y_train, k = 15)


# In[145]:


n_test = x_test.shape[0]
accuracy = 0

for i in range(n_test):
    y_hat = model.predict(x_test[i], distance_fn = Distance.Euclidean)
    accuracy += (y_hat == y_test[i])/n_test    
    
accuracy = round(accuracy, 3)
print(f'Accuracy : {accuracy}')


# In[146]:


y_ = [5.2, 3.1, 1.4, 0.2]
model.predict(y_, distance_fn = Distance.Euclidean)


# In[ ]:




