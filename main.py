# Made by: Carlos Leon, Ryan Arjun, Subhrajyoti Pradhan
# Further Development by Carlos Leon

## Import required libraries
import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier

## Import the 5 feature selection algorithms
import varience_threshold as vt
import tree_selection_features as tsf
import recursive_features as rf
import chi_square as cs
import information_gain as ig
import k_selection as k

## Import outliers detection system and performance
import performance as p
import outlier_detection_system as ods

## GLobal Variables
debug = "=" * 30 + "\n"                                                  
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
if len(sys.argv) != 3:
    print("Please enter commands as such\n./main.py number_of_runs(integer) feature selection algorithms to use(string)")
    print("\tv=variance")
    print("\tt=tree")
    print("\tr=recursive")
    print("\tm=mutual")
    print("\tc=chi")
    print("\tex) ./main.py 10 vtr\n\n")
    sys.exit(1)

debug += "The arguments are: " + str(sys.argv) + "\n"
_number_of_runs = int(sys.argv[1])
_use_varience_algo_ = "v" in sys.argv[2]
_use_chi_algo = "c" in sys.argv[2]
_use_recursive_algo_ = "r" in sys.argv[2]
_use_tree_algo_ = "t" in sys.argv[2]
_use_info_algo = "m" in sys.argv[2]
if not _use_info_algo and not _use_tree_algo_ and not _use_recursive_algo_ and not _use_chi_algo and not _use_varience_algo_:
    print(" No algorithms chosen")
    sys.exit(1)
else:
    print("Starting")

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
debug += "Total number of users: " + str(len(files)) + "\n"

## Start loop of runs
for i in range(_number_of_runs):
    debug += "-" * 25 + "\n"
    debug += "Beginning Run: " + str(i) + "\n"

    ## Perform feature selection
    selected_features = []
    if _use_varience_algo_:
        temp, debug = vt.get_features(raw_data, raw_data_ids, debug=debug, run=i)
        selected_features.append(temp)
    if _use_tree_algo_:
        temp, debug = tsf.get_features(raw_data, raw_data_ids, debug=debug, run=i)
        selected_features.append(temp)
    if _use_recursive_algo_:
        temp, debug = rf.get_features(raw_data, raw_data_ids, debug=debug, run=i)
        selected_features.append(temp)
    if _use_chi_algo:
        temp, debug = cs.get_features(raw_data, raw_data_ids, debug=debug, run=i)
        selected_features.append(temp)
    if _use_info_algo:
        temp, debug = ig.get_features(raw_data, raw_data_ids, debug=debug, run=i)
        selected_features.append(temp)

    ## Take the intersection of the features
    features = set(selected_features[0])
    for i in range(1, len(selected_features)):
        features.intersection(selected_features[i])

    ## Remove the unused features from raw_data
    refined_data = pd.DataFrame(raw_data)
    refined_ids = raw_data_ids
    for i in reversed(range(len(raw_data.columns))):
        if i not in features:
            col = raw_data.columns[i]
            refined_data = refined_data.drop(columns=col, axis=1)
    debug += "Remaining number of columns post intersection: " + str(len(refined_data.columns)) + " columns\n"

    ## Select best K value
    k_neighbors, metric, debug = k.select_k(raw_data, raw_data_ids, debug=debug, run=i)
    
    ## Perform cross validation
    kf = KFold(n_splits=k_folds, shuffle=True)
    scaler = MinMaxScaler()
    clf = KNeighborsClassifier(n_neighbors=k_neighbors, metric=metric)

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
        print(debug)
        template, template_ids, debug = ods.remove_outliers(template, template_ids, debug=debug, run=i)

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
            confidence = max(confidence)

            # Record
            if prediction[0] == query_label:
                accuracy += 1
                genuine_scores.append(confidence)
            else:
                impostor_scores.append(confidence)
    
        # record accuracy over fold
        debug += 'Fold accuracy: ' + str(accuracy / len(test)) + "\n" 
        total_accuracy += accuracy / len(test)

    ## Plot results
    total_accuracy = total_accuracy / k_folds
    debug += "Total accuracy: " + str(total_accuracy)

    # Record/Output Data, Courtesy of Dr. Tempest Neil
    p.perf_main(genuine_scores, impostor_scores, run=i)

    # Record selected columns
    debug += "\nfeatures Selected:\n"
    index = 0
    for i in raw_data.columns:
        debug += str(index) + ": " + i + "\n"
        index += 1
    debug += "\n"

    # print running debug
    print(debug)

# Write out results
debug += "=" * 30 + "\n\n"
f = open("./RESULTS/results.txt", 'a+')
f.write(debug)
f.close()
