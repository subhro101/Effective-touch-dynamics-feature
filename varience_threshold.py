# File for the variance threshold feature selection algorithm
from sklearn.feature_selection import VarianceThreshold

# The function which will be called
def get_features(raw_data, raw_data_ids, threshold=0.10):
    '''
    Perform feature selection using variance threshold
    Defaults to 0.0, e.g. only remove features with the same value in all samples
    '''
    
    #Performs feature selection by removing features with low variance Returns the ideal columns of size the number
    sel = VarianceThreshold(threshold=(threshold * (1 - threshold)))
    sel.fit(raw_data)

     # Set aside correct columns
    return_columns = []
    index = 0
    for feature in sel.get_support():
        if feature:
            return_columns.append(index)
        index += 1

    # return
    print("VERIENCE THESHOLD: Suggesting: ", len(return_columns), " columns out of ", len(raw_data.columns))
    return return_columns
