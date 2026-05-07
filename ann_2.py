# Practical: AND-NOT Function using McCulloch-Pitts Neural Network

# McCulloch-Pitts Neuron for AND-NOT function

# Function definition
def and_not(x1, x2):
    
    # Weights
    w1 = 1      # weight for x1
    w2 = -1     # inhibitory weight for x2
    
    # Threshold
    theta = 1
    
    # Net input
    net = (x1 * w1) + (x2 * w2)
    
    # Activation function
    if net >= theta:
        return 1
    else:
        return 0

# Input combinations
inputs = [(0,0), (0,1), (1,0), (1,1)]

# Display output
print("x1  x2  Output")

for x1, x2 in inputs:
    output = and_not(x1, x2)
    print(x1, " ", x2, "   ", output)






















# ✅ Theory
# The McCulloch-Pitts neuron is one of the earliest models of artificial neurons and forms the foundation of neural network concepts.

# It is a binary threshold-based model that takes multiple binary inputs and produces a single binary output. The neuron computes a
# weighted sum of inputs and compares it with a predefined threshold to determine the output.

# In the ANDNOT function, the neuron performs a logical operation where the output is 1 only when the first input is 1 and the second
# input is 0. 

# This is achieved by assigning appropriate weights: a positive weight for the excitatory input and a negative weight for
# the inhibitory input. If the weighted sum meets or exceeds the threshold, the neuron activates and produces output 1; otherwise, it
# outputs 0.

# This model demonstrates how logical operations can be implemented using simple neural structures. Although it does not involve learning,
# it provides a strong conceptual understanding of how neurons process inputs and make decisions.

# 🔹 Key Points to Write
# Binary threshold neuron model
# Uses weights and threshold
# Implements logical functions
# ANDNOT = x1 AND (NOT x2)
# No learning mechanism

# ✅ Advantages
# Simple and easy to implement
# Useful for understanding neural concepts
# Works well for logical operations

# ❌ Disadvantages
# Limited to binary inputs and outputs
# Cannot learn from data
# Not suitable for complex problems
# 📌 Applications
# Digital logic circuits
# Basic neural network modeling
# Logical gate implementation