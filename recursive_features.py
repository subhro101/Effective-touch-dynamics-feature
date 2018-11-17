# File for the recursive feature elimination algorithm
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# The function which will be called
def get_features(raw_data, raw_data_ids):
    '''
    Performs feature selection using recursive feature
    elimination. Returns the ideal columns of size the number
    pf features
    '''

    # Define our estimator as a support vector machine
    svm = SVC(kernel="linear")

    # Set aside some data for training and some for testing

    # instantiate our eliminator
    eliminator = RFECV(estimator=svm, cv=StratifiedKFold(n_splits=2, shuffle=True), scoring='accuracy')
    eliminator.fit(raw_data, raw_data_ids)

    # Set aside correct columns
    return_columns = []
    index = 0
    for feature in eliminator.support_:
    	if feature:
    		return_columns.append(index)
    	index += 1

    # return
    print("RECUSRIVE FEATURE ELIMINATION: Suggesting: ", eliminator.n_features_, " columns out of ", (len(raw_data[0]))) 
    return return_columns
