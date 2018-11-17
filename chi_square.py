# File for the chi square feature selection algorithm
import pandas as pd
from scipy.stats import chi2_contingency

# The function which will be called
def get_features(raw_data, raw_ids, alpha=0.33):
	'''
	This function will take in the raw data and the correct label for each row
	and compute which columns are not needed in predicting the correct person.
	Will return the names of the needed columns
	'''

	# Create data frame from data
	df = pd.DataFrame(raw_data)
	df["person"] = raw_ids

	# For each column in the data frame
	return_columns = []
	for column in df:
		# Calculate statistics
		X = df[column].astype(str)
		Y = df["person"].astype(str)
		df_observed = pd.crosstab(X, Y) 
		chi2, p, dof, expected = chi2_contingency(df_observed.values)

		# Decide to keep column
		if p < alpha and column != "person":
			return_columns.append(column)

	# return
	print("CHI SQUARED: Suggesting: ", len(return_columns), " columns out of ", (len(df.columns) - 1))
	return return_columns
