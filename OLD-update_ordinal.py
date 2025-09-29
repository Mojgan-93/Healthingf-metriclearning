import os
import sys
sys.path.append(os.getcwd())

import argparse
import time

import numpy as np

import sklearn.linear_model
import sklearn.metrics
import sklearn.model_selection
import sklearn.ensemble

import preprocess_car
import preprocess_cancer
import preprocess_student
import preprocess_mimic

import expected
import objective
import optimisation
import visualization
from helpers import PlotROCCurve

from common import *
from preprocess_mimic import *

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()


def ModelAndPredict(X_train, y_train, X_test, y_test, model):

	model.fit(X_train, y_train.iloc[:, 0])
    
	if hasattr(model, 'predict_proba'):
		probs = model.predict_proba(X_test)[:, 1]  # Get probabilities if available (for classifiers)
	else:
		probs = model.predict(X_test)  # If it's a regressor, use predict directly
    
	auc = sklearn.metrics.roc_auc_score(y_test.iloc[:, 0], probs)
    
	result = {
		'y_test' : y_test,
		'probs'  : probs,
		'AUC'    : auc}
	
	return result

@time_function
def Analysis(dataset):
	# argument: --test
	parser = argparse.ArgumentParser(description = 'Process some arguments.')
	parser.add_argument('--test', action = 'store_true', help = 'An optional test flag')
	args = parser.parse_args()

	dataset_preprocessing_map = {
		'car': 		preprocess_car.PrepareCarData,
		'cancer': 	preprocess_cancer.PrepareCancerData,
		'student': 	preprocess_student.PrepareStudentData,
		'mimic': 	preprocess_mimic.PrepareMimicData
	}

	if dataset in dataset_preprocessing_map:
		X, y = dataset_preprocessing_map[dataset]()
	
		try:
			# Define output directory for plots
			output_dir = os.getcwd()  # Current working directory
			os.makedirs(output_dir, exist_ok=True)

			X, y = PrepareMimicData()

        # Display the subset
			print("Features (X):")
			print(X)
			print("Target (y):")
			print(y)
		except ValueError as e:
			print(f"Error: {e}")

	if args.test:
		
		train_size = 10
		test_size = train_size
		
	else:
		
		train_size = TRAIN_SIZE
		test_size = None

	# X_train, X_test, y_train, y_test = TrainTestSplit(X, y, train_size = train_size, test_size = test_size, random_state = 87)
	X_train = X.copy()
	y_train = y.copy()
	

	start_time = time.time()
	optimized_result = optimisation.GetOptimizedM(X_train,y_train)
	m_optimized_raw = optimized_result['m_optimized_raw']
	
	PrintEndline()
	print('optimized m = [')
	
	for sublist in m_optimized_raw:
		
		rounded_sublist = ['{:.2f}'.format(num) for num in sublist]
		print('[', ' '.join(rounded_sublist), ']')
		
	print(']')
	
	PrintEndline()
	print(f'there are {len(X)} records in train data.')
	start_time = PrintDoingTime(start_time, 'total metrics learning')
	PrintEndline()
	
	
	# 4 modelling and prediction: after optimization
	
	# 4.1 update X_train & X_test through m_optimized_raw
	X_train_ordinal_updated = expected.GetEZDataFrame(m_optimized_raw, X)
	

	# Step 2: Concatenate the updated ordinal columns back into the processed data


	### Scalling -->standard Ssalar


	## Save Train and Test
	X_train_ordinal_updated.to_csv(f"plots/XTrain_{TRAIN_SIZE}_{dataset}.csv", index=False)
	y.to_csv(f"plots/yTrain_{TRAIN_SIZE}_{dataset}.csv", index=False)

	# 3.3 get new result
	

	
	# 4 more reporting
	
	PrintEndline()
	print('train objective initial      = ', round(optimized_result['objective_value_initial'],  2))
	print('train objective intermediate = ', round(optimized_result['objective_value_intermed'], 2))
	print('train objective optimized    = ', round(optimized_result['objective_value_final'],    2))

	
	new_ordinal = visualization.Visulize_ordinal(X_train, m_optimized_raw)
	visualization.plot_value_mappings(new_ordinal)


if __name__ == '__main__':

	
	Analysis(dataset='mimic')
