#!/usr/bin/env python
# coding: utf-8

# In[2]:


from sklearn.neural_network import MLPClassifier


# In[4]:


from sklearn.neural_network import MLPClassifier
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Define the architecture of the MLP
input_size = 4
hidden_size = 5
output_size = 3

# Manually set initial weights and biases
initial_weights_hidden = np.random.rand(input_size, hidden_size)
initial_biases_hidden = np.random.rand(hidden_size)

initial_weights_output = np.random.rand(hidden_size, output_size)
initial_biases_output = np.random.rand(output_size)

# Create and configure the MLPClassifier with fixed initial weights and biases
mlp = MLPClassifier(hidden_layer_sizes=(hidden_size,),
                    max_iter=1000,
                    random_state=42,
                    warm_start=True,  # Warm start allows incremental training
                    alpha=0,  # Set regularization strength to 0 to avoid regularization
                    solver='sgd',  # Use stochastic gradient descent
                    learning_rate_init=0.01)  # Set initial learning rate

# Set initial weights and biases
mlp.coefs_ = [initial_weights_hidden, initial_weights_output]
mlp.intercepts_ = [initial_biases_hidden, initial_biases_output]

# Example input data (you would replace this with your actual input data)
X = np.random.rand(100, input_size)
y = np.random.randint(0, 2, 100)  # Binary classification for demonstration purposes

# Train the model with fixed initial weights
mlp.fit(X, y)

# Make predictions
predictions = mlp.predict(X)

# Evaluate the accuracy
accuracy = np.mean(predictions == y)
print(f"Accuracy: {accuracy:.2f}")


# In[9]:


from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Assuming you have your own dataset
# X, y = your_data_preparation_function()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features by removing the mean and scaling to unit variance
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create an MLPClassifier with verbose set to True
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, activation='relu', random_state=42, verbose=True)

mlp.coef

# Fit the model to the training data
mlp.fit(X_train, y_train)

# Make predictions on the testing data
y_pred = mlp.predict(X_test)

# Evaluate the accuracy of the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

# Print weights after each iteration
for i, weights in enumerate(mlp.coefs_):
    print(f"\nWeights after iteration {i + 1}:\n{weights}")


# In[4]:


INPUT_LAYER_NODES = 3
HIDDEN_LAYER_NODES = 2
OUTPUT_LAYER_NODES = 2


# In[6]:


mlp = MLPClassifier(hidden_layer_sizes = (HIDDEN_LAYER_NODES,),
                    activation = 'logistic',
                    max_iter = 2,
                    warm_start = True,
                    alpha = 0,
                    solver = 'sgd',
                    learning_rate_init = 0.1
                   )


# In[7]:


mlp.coefs_ = [[[0.2, 0.3, -0.4], [0.2, 0.3, -0.1]], [[0.2, -0.3], [0.4, -0.1]]]
mlp.intercepts_ = [[0.2, 0.1], [0.2, -0.1]]


# In[12]:


X = [[1, 1, 0]]
y = [[1, 0]]


# In[13]:


mlp.fit(X, y)


# In[3]:


get_ipython().run_line_magic('pinfo', 'MLPClassifier')


# In[24]:


import numpy as np


# In[13]:


def sigmoid(x):
    return 1/(1 + np.exp(-x))


# In[154]:


class Input:
   def __init__(self, n_inputs):
       self.n_inputs = n_inputs;
   
   def forward(self, x):
       return x;
   


# In[163]:


class Hidden:
   def __init__(self, n_inputs, n_outputs, weights, bias):
       self.n_inputs = n_inputs
       self.n_outputs = n_outputs
       self.weights = weights
       self.bias = bias
   
   def forward(self, x):
       self.out = []
       
       for weight in self.weights:
           self.out.append(np.dot(weight, x))
           
       self.out = sigmoid(np.array(self.out) + self.bias)
       return self.out
   
   def backward(self, x, grads):
       self.gradients = self.out * (1 - self.out) * ()


# In[164]:


class Output:
    def __init__(self, n_inputs, n_outputs, weights, bias):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.weights = weights
        self.bias = bias
        
    def forward(self, x, y):
        self.y = y
        self.out = []
        
        for weight in self.weights:
            self.out.append(np.dot(weight, x))
            
            
        self.out = sigmoid(np.array(self.out) + self.bias)
        return self.out
    
    
    def backward(self):
        self.gradients = self.out * (1 - self.out) * (self.out - y) 
        


# In[165]:


class Model:
    def __init__(self):
        self.layers = []
    
    def add(self, layer):
        self.layers.append(layer)
        
        
    def forward(self, X, y):
        out = self.layers[0].forward(X)

        for layer in self.layers[1:-1]:
            out = layer.forward(out)
        out = self.layers[-1].forward(out, y)
                    
        return out


# In[189]:


X = [0.35, 0.9]
y = [0.5]

hidden_weights = np.array([[0.1, 0.4], [0.8, 0.6]])
hidden_bias = np.zeros(2)
out_weights = np.array([[0.3, 0.9]])
out_bias = np.zeros(1)


# In[190]:


model = Model()


# In[191]:


model.add(Input(2))
model.add(Hidden(2, 2, hidden_weights, hidden_bias))
model.add(Output(2, 1, out_weights, out_bias))


# In[192]:


np.array([1, 2, 3]) * np.array([1, 5, 4])


# In[193]:


model.forward(X, y)


# In[194]:


np.dot(X, hidden_weights)


# In[195]:


from sklearn.neighbors import KNeighborsClassifier


# In[196]:


get_ipython().run_line_magic('pinfo', 'sklearn.DistanceMetric')


# In[197]:


knn = KNeighborsClassifier()


# In[198]:


import sklearn


# In[199]:


get_ipython().run_line_magic('pinfo', 'sklearn.neighbors.KNeighborsClassifier.DistanceMetric')


# In[202]:


model.layers[-1].backward()


# In[203]:


model.layers[-1].gradients


# In[205]:


import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def forward_propagation(inputs, weights):
    return sigmoid(np.dot(inputs, weights))

def train_mlp(inputs, outputs, learning_rate, epochs):
    input_size = len(inputs[0])
    output_size = len(outputs[0])
    weights = np.random.rand(input_size, output_size)

    for epoch in range(epochs):
        for i in range(len(inputs)):
            input_data = np.array(inputs[i])
            output_data = np.array(outputs[i])

            # Forward Propagation
            prediction = forward_propagation(input_data, weights)

            # Backpropagation
            error = output_data - prediction
            adjustment = learning_rate * np.dot(input_data.reshape(input_size, 1), error.reshape(1, output_size))
            weights += adjustment

        print(f"Epoch {epoch + 1}, Updated Weights: \n{weights}")

    return weights

# Hardcoded example
input_data = [
    [0, 0],
    [0, 1],
    [1, 0]
]

output_data = [
    [0],
    [1],
    [1]
]

learning_rate = 0.1
epochs = 1000

trained_weights = train_mlp(input_data, output_data, learning_rate, epochs)
print("Final Trained Weights: \n", trained_weights)


# In[ ]:




