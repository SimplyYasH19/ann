import numpy as np

# Input (ASCII 0–9)
X = np.array([48,49,50,51,52,53,54,55,56,57])

# Target: Even=1, Odd=0
Y = np.array([1,0,1,0,1,0,1,0,1,0])

# Initialize
w = 0
b = 0
lr = 0.01

# Training
for epoch in range(10):
    for i in range(len(X)):
        net = w * X[i] + b
        output = 1 if net >= 0 else 0
        error = Y[i] - output

        # Update rule
        w += lr * error * X[i]
        b += lr * error

# Testing
print("ASCII  Number  Prediction")

for x in X:
    net = w * x + b
    output = 1 if net >= 0 else 0
    number = int(chr(x))

    result = "Even" if output == 1 else "Odd"

    print(x, "   ", number, "   ", result)




















# ✅ Theory
# The perceptron is a fundamental supervised learning algorithm used for binary classification tasks. It is considered the simplest
# form of artificial neural network and consists of input nodes, weights, a bias term, and an activation function. 
# The perceptron learns by adjusting its weights based on the error between predicted and actual output using a learning rule.

# In this assignment, the perceptron is used to classify numbers as even or odd based on their ASCII values. Each input is converted
# into its numerical representation, and the perceptron processes this input to produce a binary output indicating whether the number
# is even or odd. During training, the weights are updated iteratively to minimize classification errors.

# The perceptron learning rule updates weights as:
# w = w + (learning_rate × error × input)
# This allows the model to gradually improve its predictions over time.

# However, the perceptron can only classify linearly separable data, meaning it cannot solve problems where classes are not separable
# by a straight line.

# 🔹 Key Points to Write
# Supervised learning algorithm
# Binary classification model
# Uses weight update rule
# Based on linear decision boundary
# Learns from training data

# ✅ Advantages
# Simple and fast training
# Easy to implement
# Works well for linear problems

# ❌ Disadvantages
# Cannot solve non-linear problems
# Limited to binary classification
# Sensitive to input scaling

# 📌 Applications
# Spam detection
# Pattern recognition
# Basic classification systems
