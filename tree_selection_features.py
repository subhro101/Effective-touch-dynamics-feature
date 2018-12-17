# File for the Tree Based feature selection algorithm
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import matplotlib.pyplot as plt
import numpy as np

# The function that will be called
def get_features(raw_data, raw_data_ids, debug, run, show=True):
    '''
    Uses tree selection to determine feature importance
    then used the avg of the importance to remove unneeded features
    '''

    # Create a tree classifier
    clf = ExtraTreesClassifier(n_estimators=100)
    clf.fit(raw_data, raw_data_ids)

    # Calculate feature importance
    model = SelectFromModel(clf, prefit=True)

    # Set aside correct columns
    return_columns = []
    index = 0
    for feature in model.get_support():
    	if feature:
    		return_columns.append(index)
    	index += 1

    # Save results
    importances = clf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in clf.estimators_], axis=0)
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(16,8))
    plt.title("Feature importances")
    plt.bar(range(raw_data.shape[1]), importances[indices], color="r", yerr=std[indices], align="center")
    plt.xticks(range(raw_data.shape[1]), indices)
    plt.xlim([-1, raw_data.shape[1]])
    plt.savefig('./RESULTS/IMAGES/tree_feature_importance' + str(run) + '_.png', bbox_inches='tight')

    # show
    if show:
        plt.show()

    # return
    debug += "TREE BASED SELECTION: Suggesting: " + str(len(return_columns)) + " columns out of " + str(len(raw_data.columns))
    return return_columns, debug
