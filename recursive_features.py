# File for the recursive feature elimination algorithm
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
import matplotlib.pyplot as plt
import numpy as np

# The function which will be called
def get_features(raw_data, raw_data_ids, debug=1):
    '''
    Performs feature selection using recursive feature
    elimination. Returns the ideal columns of size the number
    pf features
    '''

    # Define our estimator as a support vector machine
    ls = SGDClassifier(max_iter=1000)

    # instantiate our eliminator
    eliminator = RFECV(estimator=ls, cv=StratifiedKFold(n_splits=2, shuffle=True), scoring='accuracy')
    eliminator.fit(raw_data, raw_data_ids)

    # Check if just one feature left, try again if so
    while 3 > len(eliminator.support_):
        eliminator.fit(raw_data, raw_data_ids)
        print("Running again")

    # Set aside correct columns
    return_columns = []
    index = 0
    for feature in eliminator.support_:
        if feature:
            return_columns.append(index)
        index += 1

    # DEBUG
    if debug == 1:
        plt.figure()
        plt.xlabel("Number of features selected")
        plt.ylabel("Cross validation score (nb of correct classifications)")
        plt.plot(range(1, len(eliminator.grid_scores_) + 1), eliminator.grid_scores_)
        plt.savefig('./RESULTS/IMAGES/recursive_feature_importance.png', bbox_inches='tight')
        plt.show()

    # return
    print("RECUSRIVE FEATURE ELIMINATION: Suggesting: ", eliminator.n_features_, " columns out of ", (len(raw_data.columns))) 
    return return_columns
