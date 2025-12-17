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
import preprocess_mimic_general

import expected
import objective
import optimisation
import visualization
from helpers import PlotROCCurve

from common import *
from preprocess_mimic_general import *

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
	
	# full data
	dataset_preprocessing_map = {
		'car': 		preprocess_car.PrepareCarData,
		'cancer': 	preprocess_cancer.PrepareCancerData,
		'student': 	preprocess_student.PrepareStudentData,
		'mimic': 	preprocess_mimic_general.PrepareMimicData
	}

	# Step 2: Fetch the correct dataset using the dataset argument
	if dataset in dataset_preprocessing_map:
		X, y = dataset_preprocessing_map[dataset]()
		models = [
        ('Logistic Regression', sklearn.linear_model.LogisticRegression(max_iter=1000)),
        ('Lasso Regression', sklearn.linear_model.LogisticRegression(penalty='l1', solver='liblinear', max_iter=1000)),
        ('Ridge Regression', sklearn.linear_model.LogisticRegression(penalty='l2', max_iter=1000)),
        ('Random Forest', sklearn.ensemble.RandomForestClassifier(n_estimators=100, random_state=87)),
		('GradientBoosting', sklearn.ensemble.GradientBoostingClassifier(random_state=87)),
		('Support Vector Machine', sklearn.svm.SVC(kernel='linear', random_state=87))
		# ('XGBoost', xgboost.XGBClassifier(random_state=87))
    ]

	# Create empty DataFrames to store the results
	new_results = pd.DataFrame(columns=['Model', 'AUC'])
	
	# 1 prepare data
	
	if args.test:
		
		train_size = 10
		test_size = train_size
		
	else:
		
		train_size = TRAIN_SIZE
		test_size = None
	
	#### *********** Determine X_train, X_test, y_train, y_test  BY Yourself
	train = pd.read_csv(../train.csv')
	test  = pd.read_csv('../test.csv')
	X_train = train[variable].copy()
	y_train = train[outcome].copy()
	X_test = test[variable].copy()
	y_test = test[outcome].copy()
	

	y_train_tmp = y_train.iloc[:, 0]
	print(f'y_train: total = {len(y_train_tmp)}, case = {y_train_tmp.sum()}, control = {(y_train_tmp == 0).sum()}, prev = {round(y_train_tmp.sum() / len(y_train_tmp), 2)}')
	print('\nX_test:  ', X_test.shape)
	y_test_tmp = y_test.iloc[:, 0]
	print(f'y_test: total = {len(y_test_tmp)}, case = {y_test_tmp.sum()}, control = {(y_test_tmp == 0).sum()}, prev = {round(y_test_tmp.sum() / len(y_test_tmp), 2)}')
	PrintEndline()
	
	
	# 2 modelling and prediction baseline

	X_train_processed, X_test_processed = preprocess_mimic_general.preprocess_mimic_data(X_train, X_test, y_train, y_test)

	for model_name, model in models:
		base_results = ModelAndPredict(X_train_processed, y_train, X_test_processed, y_test, model)
		probs_base = base_results['probs']
		result_baseline = PlotROCCurve(probs_base, y_test.iloc[:, 0], model_name, ci=confidence_interval, random_seed=random_seed)
		results_baseline = [model_name]
		results_baseline.extend(result_baseline)
		result_list_baseline.append(results_baseline)

	result_baseline_df = pd.DataFrame(result_list_baseline, columns=['Model', 'auroc', 'ap', 'sensitivity', 'specificity', 'threshold', 
                                               'lower_auroc', 'upper_auroc', 'std_auroc', 'lower_ap', 'upper_ap', 
                                               'std_ap', 'lower_sensitivity', 'upper_sensitivity', 'std_sensitivity',
                                               'lower_specificity', 'upper_specificity', 'std_specificity'])
	result_baseline_df.to_csv(f"plots/baseline_{TRAIN_SIZE}_{dataset}.csv", index=False)
	
	
	ordinal_columns = ['triage_acuity', 'triage_pain']
	ordinal_data_train = X_train_processed[ordinal_columns]
	ordinal_data_test = X_test_processed[ordinal_columns]
	X_train_processed_no_ordinal = X_train_processed.drop(columns=ordinal_columns)
	X_test_processed_no_ordinal = X_test_processed.drop(columns=ordinal_columns)


	start_time = time.time()
	optimized_result = optimisation.GetOptimizedM(ordinal_data_train, y_train)
	m_optimized_raw = optimized_result['m_optimized_raw']
	
	PrintEndline()
	print('optimized m = [')
	
	for sublist in m_optimized_raw:
		
		rounded_sublist = ['{:.2f}'.format(num) for num in sublist]
		print('[', ' '.join(rounded_sublist), ']')
		
	print(']')
	
	PrintEndline()
	print(f'there are {len(ordinal_data_train)} records in train data.')
	start_time = PrintDoingTime(start_time, 'total metrics learning')
	PrintEndline()

	X_train_ordinal_updated = expected.GetEZDataFrame(m_optimized_raw, ordinal_data_train)
	X_test_ordinal_updated  = expected.GetEZDataFrame(m_optimized_raw, ordinal_data_test)
	


	X_train_concat = pd.concat([X_train_processed_no_ordinal, X_train_ordinal_updated], axis=1)
	X_test_concat = pd.concat([X_test_processed_no_ordinal, X_test_ordinal_updated], axis=1)

	scaler.fit(X_train_concat)
	X_train_final = scaler.transform(X_train_concat)

	X_test_final = scaler.transform(X_test_concat)

	X_train_final.to_csv(f"plots/XTrain_{TRAIN_SIZE}_{dataset}.csv", index=False)
	y_train.to_csv(f"plots/yTrain_{TRAIN_SIZE}_{dataset}.csv", index=False)
	X_test_final.to_csv(f"plots/XTest_{TRAIN_SIZE}_{dataset}.csv", index=False)
	y_test.to_csv(f"plots/yTest_{TRAIN_SIZE}_{dataset}.csv", index=False)

	for model_name, model in models:

		new_results_data = ModelAndPredict(X_train_final, y_train, X_test_final, y_test, model)
		probs = new_results_data['probs']
		result_complete = PlotROCCurve(probs, y_test.iloc[:, 0], model_name, ci=confidence_interval, random_seed=random_seed)
		results = [model_name]
		results.extend(result_complete)
		result_list.append(results) 
		

	print("\nNew Results:")
	new_results.to_csv('new_results.csv', index=False)
	
	print(new_results)
	
	PrintEndline()
	start_time = PrintDoingTime(start_time, 'modelling and prediction after metrics learning')
	PrintEndline()
	result_df = pd.DataFrame(result_list, columns=['Model', 'auroc', 'ap', 'sensitivity', 'specificity', 'threshold', 
                                               'lower_auroc', 'upper_auroc', 'std_auroc', 'lower_ap', 'upper_ap', 
                                               'std_ap', 'lower_sensitivity', 'upper_sensitivity', 'std_sensitivity',
                                               'lower_specificity', 'upper_specificity', 'std_specificity'])
	result_df.to_csv(f"plots/metricresult_{TRAIN_SIZE}_{dataset}.csv", index=False)

	result_df = result_df.round(3)
	formatted_result_df = pd.DataFrame()
	formatted_result_df[['Model', 'Threshold']] = result_df[['Model', 'threshold']]
	formatted_result_df['AUROC'] = result_df['auroc'].astype(str) + ' (' + result_df['lower_auroc'].astype(str) + \
                               '-' + result_df['upper_auroc'].astype(str) + ')'
	formatted_result_df['AUPRC'] = result_df['ap'].astype(str) + ' (' + result_df['lower_ap'].astype(str) + \
                               '-' + result_df['upper_ap'].astype(str) + ')'
	formatted_result_df['Sensitivity'] = result_df['sensitivity'].astype(str) + ' (' + result_df['lower_sensitivity'].astype(str) + \
                                     '-' + result_df['upper_sensitivity'].astype(str) + ')'
	formatted_result_df['Specificity'] = result_df['specificity'].astype(str) + ' (' + result_df['lower_specificity'].astype(str) + \
                                     '-' + result_df['upper_specificity'].astype(str) + ')'
	formatted_result_df.to_csv(f"plots/FinalMetric_{TRAIN_SIZE}_{dataset}.csv", index=False)
	formatted_result_df	
	
	# 4 more reporting
	
	PrintEndline()
	print('train objective initial      = ', round(optimized_result['objective_value_initial'],  2))
	print('train objective intermediate = ', round(optimized_result['objective_value_intermed'], 2))
	print('train objective optimized    = ', round(optimized_result['objective_value_final'],    2))
	
	
	if args.test:
	#if True:
		
		Matrix_E_z = expected.GetEZMatrix(m_optimized_raw, ordinal_data_test)
		matrix_var = {f'matrix_{v+1}': expected.GetVarZMatrix(v, m_optimized_raw, ordinal_data_test) for v in range(len(ordinal_data_test))}
		
		PrintEndline()
		test_objective_initial   = objective.ObjectiveFunc_m(optimized_result['m'],           optimized_result['lengths_m'], optimized_result['M_psd'], optimized_result['gamma'], optimized_result['lambda_'], ordinal_data_test, y_test)
		test_objective_intermed  = objective.ObjectiveFunc_m(optimized_result['m_optimized'], optimized_result['lengths_m'], optimized_result['M_psd'], optimized_result['gamma'], optimized_result['lambda_'], ordinal_data_test, y_test)
		
		#test_objective_final     = objective.ObjectiveFunc_M(optimized_result['M_optimized_flatten'], optimized_result['Matrix_E_z'], optimized_result['matrix_var'], optimized_result['gamma'], optimized_result['lambda_'], X_test, y_test)
		test_objective_final     = objective.ObjectiveFunc_M(optimized_result['M_optimized_flatten'], Matrix_E_z, matrix_var, optimized_result['gamma'], optimized_result['lambda_'], ordinal_data_test, y_test)
		
		print('test objective initial      = ', round(test_objective_initial,  2))
		print('test objective intermediate = ', round(test_objective_intermed, 2))
		print('test objective optimized    = ', round(test_objective_final,    2))
		
		PrintEndline()
		print(f'there are {len(ordinal_data_test)} records in test data.')
		start_time = PrintDoingTime(start_time, 'test data distance calculation')
		PrintEndline()
	
	
	new_ordinal = visualization.Visulize_ordinal(ordinal_data_train, m_optimized_raw)
	visualization.plot_value_mappings(new_ordinal)



if __name__ == '__main__':

	
	Analysis(dataset='student')
