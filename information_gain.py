# File for the information gain feature selection algorithm
import numpy as np

# The function which will be called
def information_gain(data, split_attribute_name, target_name="class"):

    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """

    def entropy(target_col):
        """
        Calculate the entropy of a dataset.
        The only parameter of this function is the target_col parameter which specifies the target column
        """
        elements, counts = np.unique(target_col, return_counts=True)
        entropy = np.sum([(-counts[i]/np.sum(counts))*np
                         .log2(counts[i]/np.sum(counts)) for i in range(len(elements))])
        return entropy

    # Calculate the entropy of the total dataset
    total_entropy = entropy(data[target_name])

    # Calculate the entropy of the dataset

    # Calculate the values and the corresponding counts for the split attribute
    vals, counts= np.unique(data[split_attribute_name], return_counts=True)

    # Calculate the weighted entropy
    weighted_entropy = np.sum([(counts[i]/np.sum(counts))*entropy(
        data.where(data[split_attribute_name] == vals[i]).dropna()[target_name]) for i in range(len(vals))])

    # Calculate the information gain
    information_gain = total_entropy - weighted_entropy
    return information_gain