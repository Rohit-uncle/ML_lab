#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd


# In[15]:


def findS(data, positive_label = '1'):
    examples = data[:, :-1]
    labels = data[:, -1]

    n_features = examples.shape[1]
    
    hypothesis = ['0'] * n_features
    
    for example, label in zip(examples, labels):
        if label != positive_label:
            continue
        
        for ind, attribute in enumerate(example):
            if hypothesis[ind] == '0':
                hypothesis[ind] = attribute
            elif hypothesis[ind] != attribute:
                hypothesis[ind] = '?'
                
    return hypothesis
            


# In[16]:


data = np.array([['Morning', 'Sunny', 'Warm', 'Yes', 'Mild', 'Strong', 'Yes'],
['Evening', 'Rainy', 'Cold', 'No', 'Mild', 'Normal', 'No'],
['Morning', 'Sunny', 'Moderate', 'Yes', 'Normal', 'Normal', 'Yes'],
['Evening', 'Sunny', 'Cold', 'Yes', 'High', 'Strong', 'Yes']])


# In[17]:


findS(data, positive_label = 'Yes')


# In[ ]:




