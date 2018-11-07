# Import required libraries
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Import the 5 feature selection algorithms
#import varience_threshold as vt
#import pca as pca
#import minimum_subset as ms
#import chi_square as cs
#import information_gain as ig

# GLobal Variables
train_percent = .75
test_percent = .25
dataset_path = "dataset1/"

# Load the dataset
raw_data = []
files = os.listdir(dataset_path)
for f in files:
    temp = pd.read_csv(dataset_path + f, header=None)
    raw_data.append(temp)

# Perform Isolation Forest for outlier detection
clf = IsolationForest()
refined_data = []
for person in raw_data:
    end = ceil(len(person) * train_percent)
    train_temp = person[0:end]

    for train_row in range(end):
        test_row = train_temp[train_row]
        training_rows = [i for i in range(end) if i != test_row]
        clf.fit(training_rows)
        predict = clf.predict(test_row)
        if predict == 1:
            refined_data.append(test_row)



# Perform feature selection
varience_threshold_features = []
pca_threshold_features = []
minimum_subset_features = []
chi_square_features = []
information_gain_features = []

# Take the intersection of the features
features = set(varience_threshold_features).intersection(pca_threshold_features)
features = features.intersection(minimum_subset_features)
features = features.intersection(chi_square_features)
features = features.intersection(information_gain_features)

# Extract test, train from k-fold k=5 validation

# For each feature set

    # Train on train

    # Test on test

    # Compute results


# Plot results

# Record/Output Data
