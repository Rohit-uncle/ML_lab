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
