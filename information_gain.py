# File for the infomration gain feature selection algorithm
# https://stackoverflow.com/questions/47241525/python-information-gain-implementation
from scipy.stats import entropy
import numpy as np

def information_gain(X, y):

    def _entropy(labels):
        counts = np.bincount(labels)
        return entropy(counts, base=None)

    def _ig(x, y):
        # indices where x is set/not set
        x_set = np.nonzero(x)[1]
        x_not_set = np.delete(np.arange(x.shape[1]), x_set)

        h_x_set = _entropy(y[x_set])
        h_x_not_set = _entropy(y[x_not_set])

        return entropy_full - (((len(x_set) / f_size) * h_x_set)
                             + ((len(x_not_set) / f_size) * h_x_not_set))

    entropy_full = _entropy(y)
    f_size = float(X.shape[0])
    scores = np.array([_ig(x, y) for x in X.T])
    return scores