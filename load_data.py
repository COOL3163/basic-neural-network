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