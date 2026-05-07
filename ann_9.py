# Back Propagation Feed Forward Neural Network

import numpy as np

# Input Data
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# Target Output
Y = np.array([[0],
              [1],
              [1],
              [0]])

# Initialize weights
W1 = np.random.rand(2,2)
W2 = np.random.rand(2,1)

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative Function
def derivative(x):
    return x * (1 - x)

# Training
for i in range(5000):

    # Feed Forward

    hidden = sigmoid(np.dot(X, W1))

    output = sigmoid(np.dot(hidden, W2))

    # Error

    error = Y - output

    # Back Propagation

    d_output = error * derivative(output)

    hidden_error = d_output.dot(W2.T)

    d_hidden = hidden_error * derivative(hidden)

    # Update weights

    W2 = W2 + hidden.T.dot(d_output)

    W1 = W1 + X.T.dot(d_hidden)

# Final Output

print("Output:\n")

print(np.round(output))


























# ✅ Theory
# A feedforward neural network is the simplest type of artificial neural network in which data flows in only one 
# direction, from input layer to output layer, without any feedback loops. 
 
# It consists of an input layer, one or more hidden layers, and an output layer.
# Each neuron computes a weighted sum of inputs and applies an activation function. 

# The network learns by adjusting weights using backpropagation, which minimizes the difference between predicted and actual output.
# Unlike recurrent networks, feedforward networks do not have memory and cannot handle sequential data. 

# However, they are highly effective for classification and regression tasks where input-output relationships are direct.
# This assignment demonstrates how a neural network processes data layer by layer and learns patterns through iterative training.

# 🔹 Key Points to Write
# One-direction data flow
# No feedback loops
# Uses hidden layers
# Trained using backpropagation
# Used for classification/regression

# ✅ Advantages
# Simple and efficient
# Easy to implement
# Works well for structured data

# ❌ Disadvantages
# Cannot handle sequential data
# No memory of previous inputs
# Limited flexibility

# 📌 Applications
# Pattern recognition
# Prediction systems
# Data classification