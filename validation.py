# Made by: Carlos Leon, Ryan Arjun, Subhrajyoti Pradhan

## Import required libraries
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing
from performance import perf_main

## Import the 5 feature selection algorithms
import varience_threshold as vt
import tree_selection_features as tsf
import recursive_features as rf
import chi_square as cs
import information_gain as ig

## Import outliers detection system
import outlier_detection_system as ods

## GLobal Variables
debug = 0
subset_size = 2000
dataset_path = "dataset2/evoline1/"
k_folds = 5
k_neighbors = 30
headers = ['UserId', ' DeviceId', ' SessionId', ' Key', ' DownTime', ' UpTime',
 'Pressure', ' FingerArea', ' RawX', ' RawY', ' gravityX', ' gravityY', ' gravityZ',
' Hands', ' PasswordType', ' Repetition']

## Load the dataset
raw_data = pd.DataFrame(columns=headers, data=[])
raw_data_ids = []
ids = 0
temp_list = []
files = os.listdir(dataset_path)
for f in files:
    temp = pd.read_csv(dataset_path + f)
    temp_list.append(temp) 

raw_data = pd.concat(temp_list)

# If dataset is too big acquire a subset
if len(raw_data) > subset_size:
    raw_data = raw_data.sample(subset_size)

# Perform Label Encoding
le = preprocessing.LabelEncoder()
raw_data_ids = raw_data['UserId']
raw_data_ids= le.fit_transform(raw_data_ids)
raw_data = raw_data.drop(columns="UserId", axis=1)
raw_data[' DeviceId'] = le.fit_transform(raw_data[' DeviceId'])
raw_data[' Key'] = le.fit_transform(raw_data[' Key'])
raw_data = raw_data.astype(np.float64)
raw_data_ids = np.array(raw_data_ids)
print("Total number of raw rows: ", len(raw_data))

## Perform feature selection
def get_set_of_features():
    varience_threshold_features = vt.get_features(raw_data, raw_data_ids)
    tree_selection_features = tsf.get_features(raw_data, raw_data_ids, debug=debug)
    recusive_features = rf.get_features(raw_data, raw_data_ids, debug=debug)
    chi_square_features = cs.get_features(raw_data, raw_data_ids)
    information_gain_features = ig.get_features(raw_data, raw_data_ids)

    ## Take the intersection of the features
    features = set(chi_square_features).intersection(recusive_features)
    features = features.intersection(varience_threshold_features)
    features = features.intersection(tree_selection_features)
    features = features.intersection(information_gain_features)
    return features

# Check if less than two features selected, if so run again
features = get_set_of_features()
index = 2
while len(features) < 2:
    print("FAILED at finding intersecting features, trying again, iteration=", index)
    features = get_set_of_features()
    print("New len, ", len(features))
    index += 1


## Remove the unused features from raw_data
for i in reversed(range(len(raw_data.columns))):
    if i not in features:
        col = raw_data.columns[i]
        raw_data = raw_data.drop(columns=col, axis=1)
print("Remaining number post intersection: ", len(raw_data.columns), " columns")
    
## Perform cross validation
kf = KFold(n_splits=k_folds, shuffle=True)
scaler = MinMaxScaler()
clf = KNeighborsClassifier(n_neighbors=k_neighbors)

# Set aside performance variables
genuine_scores = []
impostor_scores = []
total_accuracy = 0.

for train, test in kf.split(raw_data, raw_data_ids):

    # Get current folds data
    accuracy = 0.
    template = raw_data.values[train, :]
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
        query = raw_data.values[test[test_index], :]
        query = query.reshape(len(query), 1)
        query = scaler.fit_transform(query)
        query = query.reshape(1, -1)
        query_label = raw_data_ids[test[test_index]]

        # Predict
        prediction = clf.predict(query)
        confidence = clf.predict_proba(query)

        # Record
        if prediction[0] == query_label:
            accuracy += 1
            genuine_scores.append(confidence[0][prediction[0]])
        else:
            impostor_scores.append(confidence[0][prediction[0]])

    # DEBUG
    if debug == 1:
        print('Fold accuracy: ' + str(accuracy / len(test))) 
            
    # record accuracy over fold
    total_accuracy += accuracy / len(test)

# Plot results
total_accuracy = total_accuracy / k_folds

# Record/Output Data
# Note which features were selected and total accuracy
f = open("./RESULTS/results_" + dataset_path + ".txt", 'a+')
index = 0
f.write("-" * 25 + "\nfeatures Selected:\n")
for i in raw_data.columns:
    f.write(str(index) + ": " + i + "\n")
    index += 1
f.write("Total accuracy: " + str(total_accuracy) + "\n")
f.write("-" * 25 + "\n\n")
f.close()

# Courtesy of Dr. Tempest Neil
perf_main(genuine_scores, impostor_scores, name="_" + dataset_path)
