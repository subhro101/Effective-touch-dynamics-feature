# File for the variance threshold feature selection algorithm
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# The function which will be called


def get_features(raw_data, raw_data_ids):
    
    #Performs feature selection by removing features with low variance Returns the ideal columns of size the number
    #pf features
    
    sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
    sel.fit(raw_data, raw_data_ids) # Defaults to 0.0, e.g. only remove features with the same value in all samples
     # Set aside correct columns
    return_columns = []
    index = 0
    for feature in sel.support_:
        if feature:
            return_columns.append(index)
        index += 1

    # return
    print("RECUSRIVE FEATURE ELIMINATION: Suggesting: ", sel.n_features_, " columns out of ", (len(raw_data[0])))
    return return_columns
