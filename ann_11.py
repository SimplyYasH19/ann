def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    intersection = max(0, x2 - x1) * max(0, y2 - y1)

    area1 = (box1[2]-box1[0])*(box1[3]-box1[1])
    area2 = (box2[2]-box2[0])*(box2[3]-box2[1])

    union = area1 + area2 - intersection

    return intersection / union

# Example bounding boxes
boxA = [1,1,4,4]
boxB = [2,2,5,5]

print("IoU:", iou(boxA, boxB))

# Metrics
TP, FP, FN = 50, 10, 5

precision = TP / (TP + FP)
recall = TP / (TP + FN)

print("Precision:", precision)
print("Recall:", recall)




























# ✅ Theory
# In object detection, evaluating model performance is crucial. Instead of only predicting class labels, object detection 
# models also predict bounding boxes. Therefore, specialized evaluation metrics are used.

# Intersection over Union (IoU) measures the overlap between predicted and actual bounding boxes. 
# It is calculated as:
# IoU = Area of Intersection / Area of Union

# A higher IoU indicates better prediction accuracy.

# Precision measures how many predicted objects are actually correct:
# Precision = TP / (TP + FP)

# Recall measures how many actual objects are correctly detected:
# Recall = TP / (TP + FN)

# These metrics help evaluate detection quality and are widely used in computer vision tasks.

# 🔹 Key Points to Write
# IoU measures overlap
# Precision measures correctness
# Recall measures completeness
# Used in object detection
# Helps evaluate model performance

# ✅ Advantages
# Standard evaluation metrics
# Provides detailed performance insight
# Helps compare models

# ❌ Disadvantages
# Depends on threshold selection
# May not reflect full performance alone
# Requires proper interpretation

# 📌 Applications
# Object detection
# Image analysis
# Computer vision systems