# Logistic Regression using TensorFlow

import tensorflow as tf
import numpy as np

# Input Data
X = np.array([[0],
              [1],
              [2],
              [3]], dtype=float)

# Output Data
Y = np.array([[0],
              [0],
              [1],
              [1]], dtype=float)

# Create Model
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1,
                          activation='sigmoid',
                          input_shape=(1,))
])

# Compile Model
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train Model
model.fit(X, Y, epochs=100)

# Evaluate Model
loss, accuracy = model.evaluate(X, Y)

print("\nAccuracy:", accuracy)

# Prediction
print("\nPredictions:")
print(model.predict(X))

































# ✅ Theory
# TensorFlow is a popular deep learning framework used to build and train machine learning models. 
# Logistic regression is a supervised learning algorithm used for binary classification problems.

# Logistic regression uses a sigmoid activation function to map input values into a probability between 0 and 1. 
# The model predicts the probability of a class and classifies it based on a threshold.

# In TensorFlow, logistic regression is implemented using layers and trained using optimization algorithms like Adam. 
# The loss function used is binary cross-entropy, which measures the difference between predicted and actual values.
# This assignment demonstrates how neural networks can be implemented using TensorFlow and how logistic regression can be 
# trained efficiently.

# 🔹 Key Points to Write
# Binary classification algorithm
# Uses sigmoid activation
# Implemented in TensorFlow
# Uses cross-entropy loss
# Uses optimization algorithms

# ✅ Advantages
# Simple and efficient
# Easy implementation
# Works well for binary problems

# ❌ Disadvantages
# Cannot handle multi-class directly
# Limited model complexity
# Sensitive to data

# 📌 Applications
# Spam detection
# Medical diagnosis
# Risk prediction