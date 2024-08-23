from __future__ import division
import cv2
import os
import numpy as np
import pickle
import matplotlib.pyplot as plt
from itertools import islice

LIMIT = None

# Reading Data from the dataset folder
DATA_FOLDER = f'{os.getcwd()}/driving_dataset'
TRAIN_FILE = os.path.join(DATA_FOLDER, 'data.txt')

# Converting images from RGB to HSV
def preprocess(img):
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:, :, 1], (100, 100))
    return resized

# Creating Labels and Features in binary format
def return_data():

    X = []
    y = []
    features = []

    with open(TRAIN_FILE) as fp:
        for line in islice(fp, LIMIT):
            path, angle = line.strip().split()
            full_path = os.path.join(DATA_FOLDER, path)
            X.append(full_path)
            # using angles from -pi to pi to avoid rescaling the atan in the network
            y.append(float(angle) * np.pi / 180)

    for i in range(len(X)):
        img = plt.imread(X[i])
        features.append(preprocess(img))

    features = np.array(features).astype('float32')
    labels = np.array(y).astype('float32')

    # Dumping binary files using pickle
    with open("features", "wb") as f:
        pickle.dump(features, f, protocol=4)
    with open("labels", "wb") as f:
        pickle.dump(labels, f, protocol=4)

return_data()