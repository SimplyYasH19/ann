# Practical: Multi-Class Classification using Neural Network from Scratch

import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder

# Load Iris Dataset
data = load_iris()

X = data.data
y = data.target.reshape(-1, 1)

# One Hot Encoding for multiclass output
encoder = OneHotEncoder(sparse_output=False)
y = encoder.fit_transform(y)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Network Architecture
input_neurons = 4
hidden_neurons = 100
output_neurons = 3

# Initialize weights and bias
np.random.seed(0)

W1 = np.random.randn(input_neurons, hidden_neurons)
b1 = np.zeros((1, hidden_neurons))

W2 = np.random.randn(hidden_neurons, output_neurons)
b2 = np.zeros((1, output_neurons))

# Activation Function - ReLU
def relu(x):
    return np.maximum(0, x)

# Derivative of ReLU
def relu_derivative(x):
    return x > 0

# Softmax Function
def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

# Loss Function
def cross_entropy(y_true, y_pred):
    m = y_true.shape[0]
    loss = -np.sum(y_true * np.log(y_pred + 1e-9)) / m
    return loss

# Training Parameters
learning_rate = 0.01
epochs = 1000

# Training Neural Network
for epoch in range(epochs):

    # Forward Propagation
    Z1 = np.dot(X_train, W1) + b1
    A1 = relu(Z1)

    Z2 = np.dot(A1, W2) + b2
    A2 = softmax(Z2)

    # Loss Calculation
    loss = cross_entropy(y_train, A2)

    # Backpropagation
    m = X_train.shape[0]

    dZ2 = A2 - y_train
    dW2 = np.dot(A1.T, dZ2) / m
    db2 = np.sum(dZ2, axis=0, keepdims=True) / m

    dA1 = np.dot(dZ2, W2.T)
    dZ1 = dA1 * relu_derivative(Z1)

    dW1 = np.dot(X_train.T, dZ1) / m
    db1 = np.sum(dZ1, axis=0, keepdims=True) / m

    # Weight Update using Gradient Descent
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1

    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    # Print loss every 100 epochs
    if epoch % 100 == 0:
        print("Epoch:", epoch, "Loss:", loss)

# Testing

# Forward pass on test data
Z1_test = np.dot(X_test, W1) + b1
A1_test = relu(Z1_test)

Z2_test = np.dot(A1_test, W2) + b2
A2_test = softmax(Z2_test)

# Predicted class
predictions = np.argmax(A2_test, axis=1)

# Actual class
actual = np.argmax(y_test, axis=1)

# Accuracy
accuracy = np.mean(predictions == actual) * 100

print("\nAccuracy:", accuracy, "%")














# ✅ Theory
# A multi-class neural network is used to classify input data into more than two categories. Unlike binary classification, where the
# output is either 0 or 1, multi-class classification involves multiple output neurons, each representing a different class.

# The network consists of an input layer, one or more hidden layers, and an output layer. Hidden layers use activation functions like
# ReLU to introduce non-linearity, while the output layer uses the Softmax function to convert outputs into probability distributions
# across multiple classes.

# The network is trained using forward propagation and backpropagation, where the error is calculated and weights are updated using
# optimization techniques such as gradient descent. 
# The use of multiple neurons in the output layer allows the model to handle complex classification tasks effectively.

# 🔹 Key Points to Write
# Multi-class classification
# Uses softmax output layer
# Hidden layers with ReLU
# Uses backpropagation
# Multiple output neurons

# ✅ Advantages
# Handles multiple classes efficiently
# High accuracy for complex tasks
# Scalable to large datasets

# ❌ Disadvantages
# Requires more computation
# Needs large training data
# Complex training process

# 📌 Applications
# Image classification
# Speech recognition
# Object recognition