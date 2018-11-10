 # File for the chi square feature selection algorithm
import pandas as pd
from scipy.stats import chi2_contingency


def chi_square(dataframe):
        df = pd.DataFrame(dataframe)
        # groupsizes = dataframe.groupby(cols.tolist).size()
        # ctsum = groupsizes.unstack(cols[0])
        return chi2_contingency(df.fillna(0))
