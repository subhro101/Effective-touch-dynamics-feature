# Made by: Carlos Leon, Ryan Arjun, Subhrajyoti Pradhan

## Import required libraries
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
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
debug = ""                                                     
dataset_path = "dataset1/data/"
k_folds = 5
headers = ['Hold .', 'Hold t', 'Hold i', 'Hold e', 'Hold Shift',
'Hold 5', 'Hold Shift.1', 'Hold Caps', 'Hold r', 'Hold o', 'Hold a',
'Hold n', 'Hold l', 'Hold Enter', 'DD ..t', 'DD t.i', 'DD i.e',
'DD e.Shift', 'DD Shift.5', 'DD 5.Shift', 'DD Shift.Caps', 'DD Caps.r',
'DD r.o', 'DD o.a', 'DD a.n', 'DD n.l', 'DD l.Enter', 'UD ..t',
'UD t.i', 'UD i.e', 'UD e.Shift', 'UD Shift.5', 'UD 5.Shift',
'UD Shift.Caps', 'UD Caps.r', 'UD r.o', 'UD o.a', 'UD a.n', 'UD n.l',
'UD l.Enter', 'Pressure .', 'Pressure t', 'Pressure i', 'Pressure e',
'Pressure Shift', 'Pressure 5', 'Pressure Shift.1', 'Pressure Caps',
'Pressure r', 'Pressure o', 'Pressure a', 'Pressure n', 'Pressure l',
'Pressure Enter', 'Size .', 'Size t', 'Size i', 'Size e', 'Size Shift',
'Size 5', 'Size Shift.1', 'Size Caps', 'Size r', 'Size o', 'Size a',
'Size n', 'Size l', 'Size Enter', 'AvH', 'AvP', 'AvA']

## Read in command line arguments
if len(sys.argv) != 9:
    print("Please enter commands as such\n./main.py number_of_runs(integer) K(integer) use_outlier_detection(boolean)")
    print("use_varience_threshold(boolean), use_chi_squared(boolean), use_recursive_feature_elimination(boolean), use_random_forest_feature_selection(boolean)")
    print("use_mutual_information(boolean)")
    print("ex) ./main.py 3 12 true true false true false true")
    sys.exit(1)

debug += "The arguments are: " + str(sys.argv) + "\n"
_number_of_runs = sys.argv[1]
_k_value = sys.argv[2]
_use_outliers_ = sys.argv[3]
_use_varience_algo_ = sys.argv[4]
_use_chi_algo = sys.argv[5]
_use_recursive_algo_ = sys.argv[6]
_use_tree_algo_ = sys.argv[7]
_use_info_algo = sys.argv[8]

## Load the dataset
raw_data = pd.DataFrame(columns=headers, data=[])
raw_data_ids = []
ids = 0
temp_list = []
files = os.listdir(dataset_path)
for f in files:
    temp = pd.read_csv(dataset_path + f, header=None, names=headers)
    temp_list.append(temp)
    raw_data_ids.extend([ids]*len(temp.index))    
    ids += 1
raw_data = pd.concat(temp_list)
raw_data = raw_data.astype(np.float64)
raw_data_ids = np.array(raw_data_ids)
debug += "Total number of raw rows: " + str(len(raw_data)) + "\n"

## Start loop of runs
for i in range(_number_of_runs):
    # Move outside stuff in here

## Perform feature selection
varience_threshold_features = vt.get_features(raw_data, raw_data_ids)
tree_selection_features = tsf.get_features(raw_data, raw_data_ids, debug=0)
recusive_features = rf.get_features(raw_data, raw_data_ids, debug=0)
chi_square_features = cs.get_features(raw_data, raw_data_ids)
information_gain_features = ig.get_features(raw_data, raw_data_ids)

## Take the intersection of the features
features = set(chi_square_features).intersection(recusive_features)
features = features.intersection(varience_threshold_features)
features = features.intersection(tree_selection_features)
features = features.intersection(information_gain_features)

## Remove the unused features from raw_data
for i in reversed(range(len(raw_data.columns))):
   if i not in features:
       col = raw_data.columns[i]
       raw_data = raw_data.drop(columns=col, axis=1)
debug += "Remaining number post intersection: " + str(len(raw_data.columns) + " columns\n")
    
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

    
    debug += 'Fold accuracy: ' + str(accuracy / len(test)) + "\n" 
            
    # record accuracy over fold
    total_accuracy += accuracy / len(test)

# Plot results
total_accuracy = total_accuracy / k_folds

# Record/Output Data
# Courtesy of Dr. Tempest Neil
perf_main(genuine_scores, impostor_scores)

# Note which features were selected and total accuracy
f = open("./RESULTS/results.txt", 'a+')
index = 0
debug += "-" * 25 + "\nfeatures Selected:\n"
for i in raw_data.columns:
    debug += ": " + i + "\n"
    index += 1
debug += "Total accuracy: " + str(total_accuracy) + "\n"
debug += "-" * 25 + "\n\n"
f.write(debug)
f.close()
