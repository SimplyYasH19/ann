# Practical: Perceptron Learning Law with Decision Regions

import numpy as np
import matplotlib.pyplot as plt

# Input dataset
# AND Gate Example

X = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

# Target Output
Y = np.array([0, 0, 0, 1])

# Initialize weights and bias
weights = np.zeros(2)
bias = 0

# Learning rate
lr = 0.1

# Number of epochs
epochs = 10

# Step Activation Function
def activation(net):
    
    if net >= 0:
        return 1
    else:
        return 0

# Training Perceptron
for epoch in range(epochs):

    print("Epoch", epoch + 1)

    for i in range(len(X)):

        # Net input
        net_input = np.dot(X[i], weights) + bias

        # Predicted output
        output = activation(net_input)

        # Error calculation
        error = Y[i] - output

        # Weight update using Perceptron Learning Law
        weights = weights + lr * error * X[i]
        bias = bias + lr * error

        print("Input:", X[i],
              "Target:", Y[i],
              "Output:", output,
              "Error:", error)

# Final weights and bias
print("\nFinal Weights:", weights)
print("Final Bias:", bias)

# Plotting Decision Regions

# Scatter plot of points
for i in range(len(X)):

    if Y[i] == 0:
        plt.scatter(X[i][0], X[i][1], marker='o')
    else:
        plt.scatter(X[i][0], X[i][1], marker='s')

# Decision Boundary
x_values = np.linspace(-0.5, 1.5, 100)

# Equation of line:
# w1*x1 + w2*x2 + b = 0

if weights[1] != 0:
    y_values = -(weights[0] * x_values + bias) / weights[1]
    plt.plot(x_values, y_values)

# Labels
plt.title("Perceptron Decision Region")
plt.xlabel("Input X1")
plt.ylabel("Input X2")

plt.grid(True)
plt.show()

























# ✅ Theory
# The perceptron learning law is a rule used to train a perceptron model by adjusting its weights based on the prediction error.
# It is an iterative process where the model learns from input-output pairs and updates its parameters to improve accuracy.

# The decision region refers to the boundary that separates different classes in the input space. In a two-dimensional case, this
# boundary is represented as a straight line that divides the data into two regions corresponding to different classes.

# During training, the perceptron calculates the output for a given input. If the output is incorrect, the weights and bias are updated
# using the learning rule. Over time, the model converges to a set of weights that correctly classify the training data.

# Graphical representation of the decision region helps visualize how the perceptron separates different classes and adjusts the boundary
# during training.

# 🔹 Key Points to Write
# Learning based on error correction
# Updates weights iteratively
# Decision boundary separates classes
# Visual representation using graph
# Works for linearly separable data

# ✅ Advantages
# Easy to visualize
# Fast convergence
# Simple implementation

# ❌ Disadvantages
# Cannot handle non-linear data
# Limited flexibility
# May not converge for complex datasets

# 📌 Applications
# Binary classification
# Data separation problems
# Machine learning basics