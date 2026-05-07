# Practical: Plot Activation Functions in Neural Networks

import numpy as np
import matplotlib.pyplot as plt

# Generate x values
x = np.linspace(-10, 10, 200)

# Activation Functions
linear = x

sigmoid = 1 / (1 + np.exp(-x))

tanh = np.tanh(x)

relu = np.maximum(0, x)

# Create plots
plt.figure(figsize=(10, 8))

# Linear Function
plt.subplot(2, 2, 1)
plt.plot(x, linear)
plt.title("Linear Function")
plt.grid(True)

# Sigmoid Function
plt.subplot(2, 2, 2)
plt.plot(x, sigmoid)
plt.title("Sigmoid Function")
plt.grid(True)

# Tanh Function
plt.subplot(2, 2, 3)
plt.plot(x, tanh)
plt.title("Tanh Function")
plt.grid(True)

# ReLU Function
plt.subplot(2, 2, 4)
plt.plot(x, relu)
plt.title("ReLU Function")
plt.grid(True)

# Adjust layout
plt.tight_layout()

# Show plots
plt.show()











# ✅ Theory
# Activation functions are mathematical functions used in artificial neural networks to determine the output of a neuron 
# based on its input.
# They introduce non-linearity into the network, which is essential because real-world data and problems are mostly non-linear in nature.

# Without activation functions, a neural network would behave like a simple linear model regardless of the number of layers, making it
# incapable of solving complex tasks such as image recognition or speech processing.

# In a neural network, each neuron computes a weighted sum of inputs and passes it through an activation function. Common activation
# functions include Linear, Sigmoid, Tanh, and ReLU. The Sigmoid function maps input values between 0 and 1, making it useful for binary
# classification. Tanh maps values between -1 and 1 and is often used in hidden layers. ReLU (Rectified Linear Unit) outputs zero for
# negative inputs and the input itself for positive values, making it computationally efficient and widely used in deep learning.

# Activation functions also play a critical role in training neural networks using backpropagation. They help in calculating gradients
# which are used to update weights and minimize error. The choice of activation function directly affects the performance, convergence
# speed, and accuracy of the model.

# 🔹 Key Points to Write
# Introduces non-linearity in neural networks
# Applied after weighted sum in neurons
# Common types: Linear, Sigmoid, Tanh, ReLU
# Helps in learning complex patterns
# Essential for deep learning models

# ✅ Advantages
# Enables solving non-linear problems
# Improves learning capability of model
# Allows deep networks to function effectively

# ❌ Disadvantages
# Some functions cause vanishing gradient problem
# ReLU may lead to dead neurons
# Selection of wrong function reduces performance

# 📌 Applications
# Image classification
# Speech recognition
# Natural language processing
# Pattern recognition