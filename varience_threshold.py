# File for the variance threshold feature selection algorithm
from sklearn.feature_selection import VarianceThreshold

# The function which will be called
def get_features(raw_data, raw_data_ids, debug, run, threshold=0.10):
    '''
    Perform feature selection using variance threshold
    Defaults to 0.0, e.g. only remove features with the same value in all samples
    '''
    
    # Returns columns that meet the amount of variance
    sel = VarianceThreshold(threshold=threshold * (1 - threshold))
    sel.fit(raw_data)

     # Set aside correct columns
    return_columns = []
    index = 0
    for feature in sel.get_support():
        if feature:
            return_columns.append(index)
        index += 1

    # return
    debug += "VarianceThreshold threshold: " + str(threshold) + "\n"
    debug += "VERIENCE THESHOLD: Suggesting: " + str(len(return_columns)) + " columns out of " + str(len(raw_data.columns)) + "\n"
    return return_columns, debug
