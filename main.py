# Made by: Carlos Leon, Ryan Arjun, Subhrajyoti Pradhan

## Import required libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

## Import the 5 feature selection algorithms
import varience_threshold as vt
import pca_features as pca
import minimum_subset as ms
import chi_square as cs
import information_gain as ig

## GLobal Variables
train_percent = .75
dataset_path = "dataset1/data/"

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

## Perform Isolation Forest for outliers detection
clf = IsolationForest(behaviour='new', contamination='auto')
refined_train = []
refined_train_ids = []
refined_query = []
refined_query_ids = []

# for every file i.e. every user
for index, person in enumerate(raw_data):
	# Get chunk of data that will be for training, rest is query
    end = np.ceil(len(person) * train_percent).astype('int')
    train_temp = person.loc[:end, :]

    # Set aside query data
    for test_row_index in range(end, len(train_temp)):
    	query_row = train_temp[test_row_index]
    	query_row = np.array(query_row).reshape(1, -1)
    	refined_query.append(query_row)
    	refined_query_ids.append(raw_data_ids[index * len(person) + test_row_index])

    for train_row_index in range(end):
    	# Get training and testing rows
        test_row = train_temp[train_row_index]
        test_row = np.array(test_row).reshape(1, -1)
        training_rows = []
        for t in range(end):
        	if t != train_row_index:
        		training_rows.append(train_temp[t])

        # Prepare and use isolation forest
        clf.fit(training_rows)
        predict = clf.predict(test_row)

        # Add row if not an outliers
        if predict == 1:
            refined_train.append(test_row)
            refined_train_ids.append(raw_data_ids[index * len(person) + train_row_index])

# # Perform feature selection
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

# Extract test, train from k-fold k=5 validation

# For each feature set

    # Train on train

    # Test on test

    # Compute results


# Plot results

# Record/Output Data
