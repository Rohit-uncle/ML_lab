#!/usr/bin/env python
# coding: utf-8

# In[28]:


import numpy as np
import matplotlib.pyplot as plt
import random
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris


# In[3]:


iris = load_iris()
X = iris['data']
y = iris['target']


# In[22]:


x_train, x_test, y_train, y_test = train_test_split(X, y, train_size = 0.8, test_size = 0.2, shuffle = True, random_state = 42)


# In[23]:


classifier = DecisionTreeClassifier(random_state = 42)


# In[24]:


classifier.fit(x_train, y_train)


# In[35]:


plt.figure(figsize = (16, 10))
plot_tree(classifier, filled = True, feature_names = iris.feature_names, class_names = iris.target_names, rounded = True)
plt.show()


# In[25]:


pred = classifier.predict(x_test)


# In[26]:


n_test = x_test.shape[0]
accuracy = np.sum(np.where(pred == y_test, 1, 0))/n_test


# In[27]:


accuracy = round(accuracy, 3)
print(f"Accuracy : {accuracy}")


# In[ ]:




