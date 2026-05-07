import numpy as np

np.random.seed(1)

X = np.array([[0,0],[0,1],[1,0],[1,1]])
Y = np.array([[0],[1],[1],[0]])

# Weights + bias
W1 = np.random.randn(2,3)
B1 = np.zeros((1,3))

W2 = np.random.randn(3,1)
B2 = np.zeros((1,1))

def sigmoid(x):
    return 1/(1+np.exp(-x))

def derivative(x):
    return x*(1-x)

# Training
for epoch in range(5000):

    # Forward
    h = sigmoid(np.dot(X,W1) + B1)
    out = sigmoid(np.dot(h,W2) + B2)

    error = Y - out

    # Backprop
    d_out = error * derivative(out)
    d_h = d_out.dot(W2.T) * derivative(h)

    W2 += 0.1 * h.T.dot(d_out)
    B2 += 0.1 * np.sum(d_out, axis=0)

    W1 += 0.1 * X.T.dot(d_h)
    B1 += 0.1 * np.sum(d_h, axis=0)

    if epoch % 1000 == 0:
        print("Epoch:", epoch, "Loss:", np.mean(error**2))

print("\nFinal Output:")
print(out)




















# ✅ Theory
# The XOR (Exclusive OR) problem is a classic example in neural networks used to demonstrate the limitation of a 
# single-layer perceptron. XOR is a non-linear problem, meaning it cannot be solved using a linear decision boundary.

# This limitation led to the development of multi-layer neural networks.
# To solve XOR, a neural network with at least one hidden layer is required. The network performs forward propagation to compute 
# outputs and uses backpropagation to update weights based on error. 

# The hidden layer enables the network to learn non-linear patterns.
# During training, the network adjusts weights iteratively using gradient descent to minimize the loss function. Over time, the model learns the XOR mapping where:
# (0,0 → 0), (0,1 → 1), (1,0 → 1), (1,1 → 0)
# This assignment demonstrates the importance of hidden layers and backpropagation in solving non-linear classification problems.

# 🔹 Key Points to Write
# Non-linear problem
# Cannot be solved by single perceptron
# Requires hidden layer
# Uses backpropagation
# Demonstrates learning capability

# ✅ Advantages
# Solves non-linear problems
# Demonstrates neural network power
# Fundamental concept in deep learning

# ❌ Disadvantages
# Requires proper tuning
# Sensitive to initial weights
# Needs training iterations

# 📌 Applications
# Logic circuit simulation
# Neural network validation
# Learning non-linear relationships