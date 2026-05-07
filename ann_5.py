# Practical: Bidirectional Associative Memory (BAM)

import numpy as np

# Input vectors (X)
X1 = np.array([1, -1, 1])
X2 = np.array([-1, 1, -1])

# Output vectors (Y)
Y1 = np.array([1, -1])
Y2 = np.array([-1, 1])

# Weight matrix calculation
W1 = np.outer(X1, Y1)
W2 = np.outer(X2, Y2)

# Total weight matrix
W = W1 + W2

print("Weight Matrix:\n")
print(W)

# Test Pattern
X_test = np.array([1, -1, 1])

print("\nInput Vector:")
print(X_test)

# Forward recall (X → Y)
Y_out = np.dot(X_test, W)

# Activation function
Y_result = np.where(Y_out >= 0, 1, -1)

print("\nAssociated Output Vector:")
print(Y_result)

# Backward recall (Y → X)
X_out = np.dot(Y_result, W.T)

X_result = np.where(X_out >= 0, 1, -1)

print("\nRecovered Input Vector:")
print(X_result)











# ✅ Theory
# Bidirectional Associative Memory (BAM) is a type of neural network that stores pairs of patterns and allows recall in both forward
# and backward directions. It is an associative memory model, meaning it can retrieve an output pattern when provided with an input
# pattern and vice versa.

# BAM uses a weight matrix that is formed by summing the outer products of input-output vector pairs. During recall, the input vector
# is multiplied with the weight matrix to produce the associated output vector. 
# Similarly, the output can be used to retrieve the original input.

# The network operates using bipolar values (-1 and +1) and applies a sign activation function to determine the final output. BAM is
# useful for pattern recognition and memory retrieval systems.
    
# 🔹 Key Points to Write
# Associative memory model
# Stores pattern pairs
# Bidirectional recall (X ↔ Y)
# Uses outer product rule
# Works with bipolar inputs
    
# ✅ Advantages
# Can recall patterns from partial input
# Bidirectional mapping
# Robust memory system

# ❌ Disadvantages
# Limited storage capacity
# Sensitive to noise
# Requires proper training data

# 📌 Applications
# Pattern recognition
# Image processing
# Memory systems