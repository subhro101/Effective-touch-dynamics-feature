## Outlier detection implemented with Isolation Forest

# Import required Libraries
import numpy as np
from sklearn.ensemble import IsolationForest

def remove_outliers(template, template_ids):

    # Set variables
    refined_data = []
    refined_ids = []
    clf = IsolationForest(behaviour='new', contamination='auto')

    # DEBUG
    print("Performing Outlier Removal\nTemplate size: ", len(template))
    count = 0

    # Perform leave out validation to remove outliers
    for leave_out in range(len(template)):

        # Get current iteration variables
        leave_out_row = template[leave_out]
        leave_out_row = leave_out_row.reshape(1,-1)

        every_other_row = []
        for i in range(len(template)):
            if i != leave_out:
                every_other_row.append(template[i])

        # Train and predict
        clf.fit(every_other_row)
        prediction = clf.predict(leave_out_row)

        # Save row if not outlier
        if prediction == 1:
            refined_data.append(template[leave_out])
            refined_ids.append(template_ids[leave_out])
            count += 1

    print("Rows Kept: ", count)

    return refined_data, refined_ids


#clf = IsolationForest(behaviour='new', contamination='auto')
#refined_train = []
#refined_train_ids = []
#refined_query = []
#refined_query_ids = []

# for every file i.e. every user
#for index, person in enumerate(raw_data):
#	# Get chunk of data that will be for training, rest is query
#    end = np.ceil(len(person) * train_percent).astype('int')
#    train_temp = person.loc[:end, :]
#
#    # Set aside query data
#    for test_row_index in range(end, len(train_temp)):
#    	query_row = train_temp[test_row_index]
#    	query_row = np.array(query_row).reshape(1, -1)
#    	refined_query.append(query_row)
#    	refined_query_ids.append(raw_data_ids[index * len(person) + test_row_index])
#
#    for train_row_index in range(end):
#    	# Get training and testing rows
#        test_row = train_temp[train_row_index]
#        test_row = np.array(test_row).reshape(1, -1)
#        training_rows = []
#        for t in range(end):
#        	if t != train_row_index:
#        		training_rows.append(train_temp[t])
#
#        # Prepare and use isolation forest
#        clf.fit(training_rows)
#        predict = clf.predict(test_row)
#
#        # Add row if not an outliers
#        if predict == 1:
#            refined_train.append(test_row)
#            refined_train_ids.append(raw_data_ids[index * len(person) + train_row_index])
