import os
import csv
import numpy as np
from sklearn.datasets import fetch_openml

# download MNist
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.values
y = mnist.target.astype(int).values  

# Binarize images (threshold at 150) 
X_bin = (X > 150).astype(np.float32)

def shift_image(image, dx, dy):
    """
    simple image shifting function
    TODO use library funciton instead for performance 
    """
    shifted_image = np.zeros_like(image)
    for x in range(28):
        for y in range(28):
            new_x = x + dx
            new_y = y + dy
            if 0 <= new_x < 28 and 0 <= new_y < 28:
                shifted_image[new_x, new_y] = image[x, y]
    return shifted_image

# Split: train 60k, test 10k
X_train = X_bin[:60000]
y_train = y[:60000]

X_test = X_bin[60000:70000]
y_test = y[60000:70000]


# each row = [label, pixel0, pixel1, ...]
with open("data/mnist_train.csv", "w", newline='') as f_train:
    writer = csv.writer(f_train)
    for label, features in zip(y_train, X_train):
        writer.writerow([int(label)] + list(features))

with open("data/mnist_test.csv", "w", newline='') as f_test:
    writer = csv.writer(f_test)
    for label, features in zip(y_test, X_test):
        writer.writerow([int(label)] + list(features))

print("Done!")
print(f"Training set: {len(X_train)} samples")
print(f"Test set: {len(X_test)} samples")