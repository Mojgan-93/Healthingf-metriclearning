import time

import numpy as np
import pandas as pd
import pandas.testing as pdt



#TRAIN_SIZE = 150
TRAIN_SIZE = 50

# Which metho to get M2
INITIAL_M  = 'Initialized M'
IDENTITY_M = 'Identity Matrix M'
CALCULATED_M = 'Calculated M'

# For generating result
confidence_interval = 95
random_seed=0
result_list = []
result_list_baseline = []
# minimum difference between m thresholds
SMALL_DIFFERENCE = 0.1

# initial threasholds as: 0.5, 1.5, 2.5, ...(True) instead of 1, 2, 3, ...(Flase)
THRESHOLD_ONE = False


def time_function(func):
	
	def wrapper(*args, **kwargs):
		
		start_time = time.time()
		print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}\n")
		
		result = func(*args, **kwargs)
		
		end_time = time.time()
		print(f"\nEnd time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
		
		elapsed_time = end_time - start_time
		hours, remainder = divmod(elapsed_time, 3600)
		minutes, seconds = divmod(remainder, 60)
		
		print(f"Total running time: {int(hours):02} hours {int(minutes):02} minutes {int(seconds):02} seconds")
		
		return result
	
	return wrapper


def PrintEndline():
	
	print('\n')


def PrintDoingTime(start_time, doing):
	
	end_time = time.time()
	
	print(f"running time for {doing}: {round(end_time - start_time, 1)} seconds")
	
	return end_time


def StandardizeDataFrame(df):
	
	means = df.mean()
	stds = df.std()
	standardized_df = (df - means) / stds
	standardized_df = standardized_df.map(lambda x: round(x, 2))
	
	return standardized_df


def get_n_unique_rows_with_index(remaining_X, feature, remaining_vals):
	"""
	Get rows of remaining_X that include only the unique values specified in remaining_vals
	for the given categorical feature. The original index of remaining_X is preserved.
	
	Arguments:
	remaining_X -- DataFrame from which rows are sampled.
	feature -- The categorical feature column name in remaining_X.
	remaining_vals -- List of unique values of the feature to include in the returned rows.
	"""
	# Filter the DataFrame to only include rows where the feature column has values in remaining_vals
	filtered_X = remaining_X[remaining_X[feature].isin(remaining_vals)]
	
	# Sample exactly one row for each unique value in remaining_vals
	sampled_rows = filtered_X.groupby(feature).apply(lambda x: x.sample(1))
	
	# Remove the multi-index created by groupby and keep the original index
	sampled_rows.index = sampled_rows.index.droplevel(0)
	
	# Ensure the correct number of rows are returned
	assert len(sampled_rows) == len(remaining_vals), f"Expected {len(remaining_vals)} rows, but got {len(sampled_rows)}."
	
	return sampled_rows



def GetSample(X, y, n, cat_features = None, random_state = None):
	"""
	Parameters:
	-----------
	X : DataFrame
		Feature matrix.
	y : Series
		Target values.
	n : int
		Number of samples to return.
	random_state : int or None, optional
		Random state to control randomness.
	cat_features : list or None, optional
		List of categorical feature names.
		If None, all features are treated as categorical.
		
	Returns:
	--------
	sampled_X : DataFrame
		Subsampled DataFrame with at least one instance of each unique value
		in the categorical features.
	sampled_y : Series
		Corresponding target values for the subsampled DataFrame.
	"""
	
	# Ensure X is a DataFrame and y is a Series
	assert isinstance(X, pd.DataFrame), "X should be a pandas DataFrame."
	assert isinstance(y, pd.Series) or isinstance(y, pd.DataFrame), "y should be a pandas Series or a DataFrame."
	
	# Assert that X and y have the same index
	assert X.index.equals(y.index), "X and y must have the same index."
	
	# Set the global random seed for reproducibility
	if random_state is not None:
		
		np.random.seed(random_state)
	
	if cat_features is None:
		
		cat_features = X.columns.tolist()
	
	# Step 1: Randomly sample `n` rows from X
	sampled_X = X.sample(n = n)
	sampled_y = y.loc[sampled_X.index]
	
	# Step 2: Check if all unique values of each cat_feature are present
	all_values_present = True
	
	for feature in cat_features:
		
		unique_vals_in_sample = set(sampled_X[feature].unique())
		unique_vals_in_X = set(X[feature].unique())
		
		if not unique_vals_in_X.issubset(unique_vals_in_sample):
			
			all_values_present = False
			break
	
	
	# Step 3: If all values are present, return the sample
	if all_values_present:
		
		return sampled_X, sampled_y
	
	
	# Step 4: Otherwise, initialize sampled_X and sampled_y from empty and implement the original logic to sample values
	sampled_X = pd.DataFrame(columns = X.columns)  # Start with an empty DataFrame
	
	if (isinstance(pd.Series, pd.Series)):
		
		sampled_y = pd.Series(dtype = y.dtype)         # Start with an empty Series
		
	else:
		
		sampled_y = pd.DataFrame(columns = y.columns)  # Start with an empty DataFrame
	
	remaining_X = X.copy()
	remaining_y = y.copy()
	
	for feature in cat_features:
		
		# Get the unique values for this feature
		unique_vals_in_sample = set(sampled_X[feature].dropna().unique()) if not sampled_X.empty else set()
		unique_vals_in_X = set(remaining_X[feature].dropna().unique())
		remaining_vals = unique_vals_in_X - unique_vals_in_sample
		
		if remaining_vals:
			
			# Sample rows corresponding to the remaining unique values for this feature # remaining_X[remaining_X[feature].isin(remaining_vals)]
			additional_samples = get_n_unique_rows_with_index(remaining_X, feature, remaining_vals)
			sampled_X = pd.concat([sampled_X, additional_samples])
			sampled_y = pd.concat([sampled_y, remaining_y.loc[additional_samples.index]])
			remaining_X = remaining_X.drop(index = additional_samples.index)
			remaining_y = remaining_y.drop(index = additional_samples.index)
	
	# If the sampled rows are less than n, sample the remaining rows to complete n
	if len(sampled_X) < n:
		
		extra_samples_X = remaining_X.sample(n = n - len(sampled_X))
		extra_samples_y = remaining_y.loc[extra_samples_X.index]
		sampled_X = pd.concat([sampled_X, extra_samples_X])
		sampled_y = pd.concat([sampled_y, extra_samples_y])
	
	if len(sampled_X) > n:
		
		print(f'warning: GetSample returns {len(sampled_X)} rows instead of {n} rows')
	
	
	return sampled_X, sampled_y


def TrainTestSplit(X, y, train_size, test_size = None, cat_features = None, random_state = None):
	
	X_train, y_train = GetSample(X, y, n = train_size, cat_features = cat_features, random_state = random_state)
	y_train['Class'] = y_train['Class'].astype('int64')
	
	X_remaining = X.drop(index = X_train.index)
	y_remaining = y.drop(index = X_train.index)
	
	if (test_size is None):
		
		return X_train, X_remaining, y_train, y_remaining
	
	
	X_test = X_remaining.sample(n = test_size)
	y_test = y_remaining.loc[X_test.index]
	
	return X_train, X_test, y_train, y_test


