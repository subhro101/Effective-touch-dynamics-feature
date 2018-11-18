# File for the information gain feature selection algorithm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif

# The function which will be called
def get_features(raw_data, raw_ids):

    """
    Calculate the information gain of a dataset. This function takes three parameters:
    Computes the avg mutual information and uses it as threshold for eliminating
    features
    """

    # Create a classifier for the data
    m_info = mutual_info_classif(raw_data, raw_ids)

    # Get the average of the mutual information of each column
    avg = np.mean(m_info)

    # Set aside correct columns
    return_columns = []
    index = 0
    for feature in m_info:
        if feature >= avg:
            return_columns.append(index)
        index += 1
    

    print("INFORMATION GAIN: Suggesting: ", len(return_columns), " columns out of ", (len(raw_data.columns)))
    return return_columns
