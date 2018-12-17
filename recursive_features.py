# File for the recursive feature elimination algorithm
from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np

# The function which will be called
def get_features(raw_data, raw_data_ids, debug, run, show=True):
    '''
    Performs feature selection using recursive feature
    elimination using a linear svm. Returns the ideal columns of size the number
    pf features
    '''

    # Define our estimator as a support vector machine
    svc = SVC(kernel="linear")

    # instantiate our eliminator
    eliminator = RFECV(estimator=svc, cv=StratifiedKFold(n_splits=2, shuffle=True), scoring='accuracy')
    eliminator.fit(raw_data, raw_data_ids)

    # Set aside correct columns
    return_columns = []
    index = 0
    for feature in eliminator.support_:
        if feature:
            return_columns.append(index)
        index += 1

    # Save results
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (nb of correct classifications)")
    plt.plot(range(1, len(eliminator.grid_scores_) + 1), eliminator.grid_scores_)
    plt.savefig('./RESULTS/IMAGES/recursive_feature_importance_' + str(run) + '.png', bbox_inches='tight')
    
    # show
    if show:
        plt.show()

    # return
    debug += "RECUSRIVE FEATURE ELIMINATION: Suggesting: " + str(eliminator.n_features_) + " columns out of " + str(len(raw_data.columns)) + "\n" 
    return return_columns, debug
