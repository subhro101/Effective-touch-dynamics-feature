# Import required libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest

# Import the 5 feature selection algorithms
import varience_threshold as vt
import pca as pca
import minimum_subset as ms
import chi_square as cs
import information_gain as ig

# Load the dataset
raw_data = []
raw_data_ids
files = os.listdir('keystroke_data/')
id_count = 0
for f in files:
    temp = pd.read_csv('keystroke_data/' + fi, header=None)
    raw_data.append(temp)
    raw_data_ids.extend([id_count] * len(temp.index))
    raw_data_ids +=1

# Perform Isolation Forest for outlier detection

# Perform feature selection
varience_threshold_features = []
pca_threshold_features = []
minimum_subset_features = []
other_algo_1_features = []
other_algo_2_features = []

# Take the intersection of the features

# Extract test, train from k-fold k=5 validation

# For each feature set

    # Train on train

    # Test on test

    # Compute results


# Plot results

# Record/Output Data
