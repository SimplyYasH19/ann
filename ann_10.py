import numpy as np

np.random.seed(1)

# New dataset (NOT XOR)
X = np.array([[1,2],
              [2,3],
              [3,4],
              [4,5]])

Y = np.array([[0],
              [0],
              [1],
              [1]])

# Weights
W1 = np.random.randn(2,3)
W2 = np.random.randn(3,1)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative(x):
    return x*(1-x)

# Training
for epoch in range(3000):

    h = sigmoid(np.dot(X,W1))
    out = sigmoid(np.dot(h,W2))

    error = Y - out

    d_out = error * derivative(out)
    d_h = d_out.dot(W2.T) * derivative(h)

    W2 += 0.1 * h.T.dot(d_out)
    W1 += 0.1 * X.T.dot(d_h)

# Output
print("Predictions:")
print(np.round(out))




























# ✅ Theory
# Hopfield network is a type of recurrent neural network used for associative memory. It stores patterns and recalls 
# them even when the input is noisy or incomplete. 

# The network consists of fully connected neurons with symmetric weights.

# The network operates using an energy function, where the goal is to minimize energy to reach a stable state. 
# During recall, the network updates neuron states iteratively until convergence.

# Hopfield networks use bipolar values (-1, +1) and are capable of storing multiple patterns. However, the number of patterns that can be stored is limited.
# This model demonstrates how neural networks can be used for memory storage and pattern retrieval.
    
# 🔹 Key Points to Write
# Recurrent neural network
# Uses energy minimization
# Stores and recalls patterns
# Works with bipolar inputs
# Fully connected network

# ✅ Advantages
# Robust to noise
# Stable recall
# Good for memory systems

# ❌ Disadvantages
# Limited storage capacity
# Slow convergence
# May produce spurious states

# 📌 Applications
# Pattern recognition
# Image reconstruction
# Optimization problems