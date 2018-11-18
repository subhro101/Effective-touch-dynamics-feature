# File for the information gain feature selection algorithm
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_selection import mutual_info_classif

# The function which will be called
def get_features(raw_data, raw_ids, average_threshold=0.5):

    """
    Calculate the information gain of a dataset. This function takes three parameters:
    1. data = The dataset for whose feature the IG should be calculated
    2. split_attribute_name = the name of the feature for which the information gain should be calculated
    3. target_name = the name of the target feature. The default for this example is "class"
    """
    df = pd.DataFrame(raw_data)
    df["person"] = raw_ids

    return_columns = []
    cv = CountVectorizer(max_df=1, min_df=1,
                         max_features=72, stop_words='english')
    for column in df:
        if column != "person":
            X = df[column].astype(str)
            Y = df["person"].astype(str)
            X_vec = cv.fit_transform(X)
            ig = mutual_info_classif(X_vec, Y, discrete_features=True)
            avg = sum(ig)
            if avg > average_threshold and column != "person":
                return_columns.append(column)

    return return_columns
