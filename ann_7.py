# ANN using Forward Propagation and Back Propagation

import numpy as np

# XOR Input
X = np.array([[0,0],
              [0,1],
              [1,0],
              [1,1]])

# XOR Output
Y = np.array([[0],
              [1],
              [1],
              [0]])

# Random weights and bias
W1 = np.random.rand(2,2)
B1 = np.random.rand(1,2)

W2 = np.random.rand(2,1)
B2 = np.random.rand(1,1)

# Sigmoid Function
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Derivative Function
def derivative(x):
    return x * (1 - x)

# Training
for i in range(5000):

    # Forward Propagation

    hidden_input = np.dot(X, W1) + B1

    hidden_output = sigmoid(hidden_input)

    final_input = np.dot(hidden_output, W2) + B2

    final_output = sigmoid(final_input)

    # Error

    error = Y - final_output

    # Back Propagation

    d_output = error * derivative(final_output)

    hidden_error = d_output.dot(W2.T)

    d_hidden = hidden_error * derivative(hidden_output)

    # Update Weights

    W2 = W2 + hidden_output.T.dot(d_output)

    W1 = W1 + X.T.dot(d_hidden)

# Output

print("Forward Propagation Output:\n")
print(np.round(final_output))

print("\nBack Propagation Error:\n")
print(error)













# ✅ Theory
# Forward propagation and backpropagation are the core processes used to train artificial neural networks. Forward propagation involves
# passing input data through the network layer by layer to compute the output. 
# Each neuron calculates a weighted sum of inputs, applies an activation function, and passes the result to the next layer.

# Backpropagation is used to minimize the error between predicted and actual output. It calculates the gradient of the loss function
# with respect to each weight using the chain rule of calculus. 
# These gradients are then used to update the weights in a way that reduces the error.

# The process is repeated over multiple iterations (epochs) until the model converges to an optimal solution. This method enables neural
# networks to learn complex patterns from data.

# 🔹 Key Points to Write
# Core training algorithm
# Uses forward and backward passes
# Minimizes error using gradients
# Uses chain rule
# Updates weights iteratively

# ✅ Advantages
# Efficient training method
# Works for deep networks
# Improves model accuracy

# ❌ Disadvantages
# Computationally intensive
# May get stuck in local minima
# Requires differentiable functions

# 📌 Applications
# Deep learning
# Neural network training
# AI systems