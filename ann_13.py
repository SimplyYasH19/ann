import tensorflow as tf

# Create CNN model
model = tf.keras.Sequential()

model.add(tf.keras.layers.Conv2D(
    32,
    (3,3),
    activation='relu',
    input_shape=(64,64,3)
))

model.add(tf.keras.layers.MaxPooling2D(
    pool_size=(2,2)
))


model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(
    128,
    activation='relu'
))

model.add(tf.keras.layers.Dense(
    1,
    activation='sigmoid'
))

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Display model structure
model.summary()






























# ✅ Theory
# Convolutional Neural Networks (CNNs) are specialized neural networks used for processing image data. 
# They automatically extract important features such as edges, textures, and patterns from images.

# A CNN consists of multiple layers including convolution layers, pooling layers, flatten layers, and fully connected layers.
# The convolution layer applies filters to extract features, while pooling reduces dimensionality and computation.

# CNNs are trained using backpropagation and are highly effective in image classification and object detection tasks.
# This assignment demonstrates the architecture and working of CNN models using TensorFlow.

# 🔹 Key Points to Write
# Used for image processing
# Uses convolution and pooling layers
# Automatic feature extraction
# Deep learning model

# ✅ Advantages
# High accuracy in image tasks
# Reduces manual feature extraction
# Handles large data efficiently

# ❌ Disadvantages
# High computational cost
# Requires large datasets
# Complex architecture

# 📌 Applications
# Image classification
# Face recognition
# Medical imaging