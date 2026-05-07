# MNIST Handwritten Digit Detection using TensorFlow
import tensorflow as tf

(X_train, Y_train), (X_test, Y_test) = \
tf.keras.datasets.mnist.load_data()

X_train = X_train / 255.0
X_test = X_test / 255.0

model = tf.keras.Sequential()

model.add(tf.keras.layers.Flatten(
    input_shape=(28,28)
))

model.add(tf.keras.layers.Dense(
    128,
    activation='relu'
))

model.add(tf.keras.layers.Dense(
    10,
    activation='softmax'
))

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(X_train, Y_train, epochs=5)

loss, accuracy = model.evaluate(X_test, Y_test)

print("\nAccuracy:", accuracy)





























# ✅ Theory
# MNIST is a widely used dataset consisting of handwritten digits from 0 to 9. 
# It is commonly used to train and test image classification models.
# In this assignment, a neural network is trained to recognize handwritten digits. 

# The input images are normalized and passed through a neural network consisting of input, hidden, and output layers.

# The output layer contains 10 neurons representing digits from 0 to 9. The model uses softmax activation to predict the probability of each digit.
# This assignment demonstrates how neural networks can be applied to real-world image classification problems.

# 🔹 Key Points to Write
# Standard image dataset
# Digit classification (0–9)
# Uses neural network
# Uses softmax activation

# ✅ Advantages
# High accuracy
# Benchmark dataset
# Easy to implement

# ❌ Disadvantages
# Requires training time
# Computational resources needed
# Limited to digit data

# 📌 Applications
# OCR systems
# Banking (cheque reading)
# Handwriting recognition