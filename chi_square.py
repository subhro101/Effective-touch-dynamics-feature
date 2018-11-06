 # File for the chi square feature selection algorithm
import pandas
from scipy.stats import chi2_contingency

class ChiSquare():

    def chi_square_dataframe_cols(self, dataframe, cols):
        groupsizes = dataframe.groupby(cols).size()
        ctsum = groupsizes.unstack(cols[0])
        return(chi2_contingency(ctsum.fillna(0)))

df = pandas.DataFrame([[0, 1, 4], [1, 0, 5], [0, 2, 6], [0, 1, 6], [0, 2, 8]], columns=['A', 'B', 'C'])
stat, p, dof, expected = ChiSquare().chi_square_dataframe_cols(dataframe=df, cols=['A', 'B', 'C'])
print('dof=%d' % dof)
print(expected)
print(stat)
print(p)