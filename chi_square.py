# File for the chi square feature selection algorithm
import pandas as pd
from scipy.stats import chi2_contingency

# The function which will be called
def get_features(raw_data, raw_ids, debug, run, alpha=0.15):
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
	index = 0
	for column in df:
		# dont check person column
		if column == "person":
			continue

		# Calculate statistics
		X = df[column].astype(str)
		Y = df["person"].astype(str)
		df_observed = pd.crosstab(X, Y) 
		chi2, p, dof, expected = chi2_contingency(df_observed.values)

		# Decide to keep column
		if p < alpha:
			return_columns.append(index)

		# update index
		index += 1

	# return
	debug += "Chi Square with alpha: " + str(alpha) + "\n"
	debug += "CHI SQUARED: Suggesting: " + str(len(return_columns)) + " columns out of " + str((len(df.columns) - 1)) + "\n"
	return return_columns, debug
