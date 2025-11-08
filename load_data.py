import os
import csv
import numpy as np
from sklearn.datasets import fetch_openml
from scipy.ndimage import shift, rotate

# download MNist
mnist = fetch_openml('mnist_784', version=1)
X = mnist.data.values
y = mnist.target.astype(int).values  

# Binarize images (threshold at 150) 
X_bin = (X > 150).astype(np.float32)

def random_shift_image(image, max_shift=2):
    image = image.reshape(28, 28)
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    shift_y = np.random.randint(-max_shift, max_shift + 1)
    shifted_image = shift(image, shift=[shift_x, shift_y], mode='constant', cval=0)
    return shifted_image.flatten()

def random_rotate_image(image, angle_range=15):
    image = image.reshape(28, 28)
    angle = np.random.uniform(-angle_range, angle_range)
    rotated_image = rotate(image, angle, reshape=False)
    return rotated_image.flatten()
    

# Split: train 50k, validation 10k, test 10k
X_train = X_bin[:50000]
y_train = y[:50000]
X_val = X_bin[50000:60000]
y_val = y[50000:60000]
X_test = X_bin[60000:70000]
y_test = y[60000:70000]

print("start")
X_augmented = []
y_augmented = []
X_augmented.extend(X_train)
y_augmented.extend(y_train)

for i in range(2):
    for j, (image, label) in enumerate(zip(X_train, y_train)):
        shifted_image = random_shift_image(image)
        X_augmented.append(shifted_image)
        y_augmented.append(label)
        rotated_image = random_rotate_image(image)
        X_augmented.append(rotated_image)
        y_augmented.append(label)

X_train = np.array(X_augmented)
y_train = np.array(y_augmented)
print("done")


# each row = [label, pixel0, pixel1, ...]
with open("data/mnist_train.csv", "w", newline='') as f_train:
    writer = csv.writer(f_train)
    for label, features in zip(y_train, X_train):
        writer.writerow([int(label)] + list(features))

with open("data/mnist_val.csv", "w", newline='') as f_val:
    writer = csv.writer(f_val)
    for label, features in zip(y_val, X_val):
        writer.writerow([int(label)] + list(features))

with open("data/mnist_test.csv", "w", newline='') as f_test:
    writer = csv.writer(f_test)
    for label, features in zip(y_test, X_test):
        writer.writerow([int(label)] + list(features))

print("Done!")
print(f"Training set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")