## Outlier detection implemented with Isolation Forest

# Import required Libraries
import numpy as np
from sklearn.ensemble import IsolationForest

def remove_outliers(template, template_ids):

    # Set variables
    refined_data = []
    refined_ids = []
    clf = IsolationForest(behaviour='new', contamination='auto')
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

    print("Rows Kept: ", count, " out of: ", len(template))
    return refined_data, refined_ids
