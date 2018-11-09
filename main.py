# Made by: Carlos Leon, Ryan Arjun, Subhrajyoti Pradhan

## Import required libraries
import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

## Import the 5 feature selection algorithms
import varience_threshold as vt
import pca_features as pca
import minimum_subset as ms
import chi_square as cs
import information_gain as ig

## Import outlier detection system
import outlier_detection_system as ods

## GLobal Variables
train_percent = .75
dataset_path = "dataset1/data/"
k_folds = 5
k_neighbors = 7

## Load the dataset
raw_data = []
raw_data_ids = []
ids = 0
files = os.listdir(dataset_path)
for f in files:
    temp = pd.read_csv(dataset_path + f, header=None)
    raw_data.append(temp)
    raw_data_ids.extend([ids]*len(temp.index))    
    ids+=1
raw_data = pd.concat(raw_data, axis=0).values
raw_data_ids = np.array(raw_data_ids)

## Perform feature selection
# varience_threshold_features = []
# pca_threshold_features = []
# minimum_subset_features = []
# chi_square_features = []
# information_gain_features = []

# # Take the intersection of the features
# features = set(varience_threshold_features).intersection(pca_threshold_features)
# features = features.intersection(minimum_subset_features)
# features = features.intersection(chi_square_features)
# features = features.intersection(information_gain_features)


## Perform cross validation
kf = StratifiedKFold(n_splits=k_folds)
scaler = MinMaxScaler()
clf = KNeighborsClassifier(n_neighbors=k_neighbors)
for train, test in kf.split(raw_data, raw_data_ids):

    # Get current folds data
    accuracy = 0.
    template = raw_data[train, :]
    template_ids = raw_data_ids[train]
    
    # Remove outliers from training data
    template, template_ids = ods.remove_outliers(template, template_ids)

    # Scale data
    template = scaler.fit_transform(template)

    # Train on data
    clf.fit(template, template_ids)

    # Test on data
    for test_index in range(len(test)):

        # Get and Scale query
        query = template[test[test_index], :].reshape(1, -1)
        query = scaler.fit_transform(query)

        # Prefict and record
        prediction = clf.predict(query)
        accuracy += prediction

# Plot results

# Record/Output Data
