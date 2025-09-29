#!/usr/bin/env python

#SBATCH --partition=physical

#SBATCH --job-name=23_features_gb
#SBATCH --time=5-00:00:00
#SBATCH --mem=128G
#SBATCH --output=/data/gpfs/projects/punim1824/%u/hpc_output/analysis_%x-%j.out
#SBATCH --error=/data/gpfs/projects/punim1824/%u/hpc_output/analysis_%x-%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=mkouhounesta@student.unimelb.edu.au


#module load anaconda3/2021.11

#eval "$(conda shell.bash hook)"

#conda activate target1
##########################

import os
import sys
sys.path.append(os.getcwd())
import time
import random
import tensorflow as tf
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import optimizers, metrics
from tensorflow.keras.optimizers import legacy

from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.layers import concatenate
from tensorflow.keras.models import load_model

##from helpers import PlotROCCurve
from mojimetrics import *
from tensorflow.keras.models import load_model
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeClassifier

from mojimetrics import *
from sklearn import metrics
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import string
import re
from sklearn.feature_extraction.text import CountVectorizer
from scipy.stats import pointbiserialr
from sklearn.feature_extraction.text import TfidfVectorizer
# from helpers import *
np.random.seed(0)
encoder = LabelEncoder()

scaler = MinMaxScaler()
result_list = []

roc_auc_list = []
average_precision_list= []
sensitivity_list= []
specificity_list = []

folder_path = '/Users/mkouhounesta/Desktop/mimic/prepared-data/5fold_with_icd'
file_extension = '.csv'
file_list = [file for file in os.listdir(folder_path) if file.endswith(file_extension)]

variables = ["age", "gender",

            "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d",
            "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d",

            "triage_temperature", "triage_heartrate", "triage_resprate",
            "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
           # "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
           # "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough",
           # "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope",
           # "chiefcom_dizziness",

            #"cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia",
            #"cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1",
            #"cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2",
            #"cci_Cancer2", "cci_HIV",

            #"eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2",
            #"eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy",
            #"eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
            #"eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression",

            "chiefcomplaint", "arrival_transport", "race"
            ]
outcome = "outcome_hospitalization"


def update_triage_columns(df):
    """
    Updates the 'triage_acuity' and 'triage_pain' columns based on predefined mappings.

    Mapping for 'triage_acuity':
    1 -> 0.13, 2 -> 1.175, 3 -> 3.795, 4 -> 5.47, 5 -> 5.57

    Mapping for 'triage_pain':
    1 -> 8.37, 2 -> 8.47, ..., 21 -> 15.98

    Parameters:
    df (pd.DataFrame): DataFrame containing 'triage_acuity' and 'triage_pain' columns.

    Returns:
    pd.DataFrame: Updated DataFrame with replaced values.
    """
    # Define mapping for 'triage_acuity'
    acuity_mapping = {
        1: 0.13, 2: 1.175, 3: 3.795, 4: 5.47, 5: 5.57
    }

    # Define mapping for 'triage_pain'
    pain_mapping = {
        1: 8.37, 2: 8.47, 3: 8.57, 4: 8.67, 5: 8.77, 6: 8.87, 7: 8.97, 8: 9.07, 9: 9.17, 10: 11.97,
        11: 14.76, 12: 14.86, 13: 14.96, 14: 15.06, 15: 15.16, 16: 15.26, 17: 15.36, 18: 15.57, 19: 15.78,
        20: 15.88, 21: 15.98
    }
    
    # Replace values in the columns if they exist
    if 'triage_acuity' in df.columns:
        df['triage_acuity'] = df['triage_acuity'].replace(acuity_mapping)

    if 'triage_pain' in df.columns:
        df['triage_pain'] = df['triage_pain'].replace(pain_mapping)

    return df


def chief_preprocess(data):
    data.loc[:, 'chiefcomplaint'] = data['chiefcomplaint'].str.lower()
    data['chiefcomplaint'].fillna('notreported', inplace=True)
    # data.loc[:, 'icd_title'] = data['icd_title'].str.lower()
    # data['icd_title'].fillna('notreportedicd', inplace=True)
    return data

def jaccard_similarity(a, b):
    if not isinstance(a, str) or not isinstance(b, str):
        return 0.0
    a_tokens = set(a.split())
    b_tokens = set(b.split())
    intersection = len(a_tokens.intersection(b_tokens))
    union = len(a_tokens.union(b_tokens))
    return intersection / union if union != 0 else 0.0

def find_nearest_strings(train_strings, test_string, k=5):
    # Check if the test_string is a valid string or NaN/float
    if not isinstance(test_string, str):
        # If the test_string is NaN or float, set similarity_scores to 0 for all training strings
        similarity_scores = [0] * len(train_strings)
    else:
    # Calculate the similarity between the test string and all strings in the training set

        similarity_scores = [jaccard_similarity(train_string, test_string) for train_string in train_strings]

    # Get indices of the k-nearest strings based on the similarity scores
    nearest_indices = np.argsort(similarity_scores)[-k:]

    return nearest_indices

def process(X):

    X['gender'] = encoder.fit_transform(X['gender'])
    X = X.fillna(X.mean())
    return X


def impute_target_encoding(train_data, test_string, target_column, k=5):

    nearest_indices = find_nearest_strings(train_data[target_column], test_string, k)
    target_values = train_data.iloc[nearest_indices]['chiefcomplaint'].values

    mean_value = np.mean(target_values)

    return mean_value


confidence_interval = 95
random_seed=0
Target_Encoder1 = TargetEncoder()
Target_Encoder2 = TargetEncoder()
Target_Encoder3 = TargetEncoder()
Target_Encoder4 = TargetEncoder()

for i in range(1, 2):
    train_path = os.path.join(folder_path, f'train_set_fold{i}.csv')
    test_path = os.path.join(folder_path, f'test_set_fold{i}.csv')

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    
    print('training size =', len(train), ', testing size =', len(test))
    X_train = train[variables]
    y_train = train[outcome]

    X_test = test[variables]
    y_test = test[outcome]

    # X_train = chief_preprocess(X_train)
    # X_test  = chief_preprocess(X_test)

    print('enjoy moji')
    X_train.loc[:, 'gender'] = encoder.fit_transform(X_train['gender'])
    X_test.loc[:, 'gender'] = encoder.transform(X_test['gender'])

    merged_data = pd.merge(X_train, y_train,  left_index=True, right_index=True)

    merged_data['outcome_encoded'] = encoder.fit_transform(merged_data['outcome_hospitalization'])
    merged_data['chiefcomplaint'] = merged_data['chiefcomplaint'].astype(str)
    ## **********************    Start New Modification       *************************
    encoded_expressions_sum = []

    # Dictionary to store mapping of expressions to their encoded values
    expression_to_encoded = {}

# Iterate through each row
    for index, row in merged_data.iterrows():
        expressions = row['chiefcomplaint'].split(',')
        encoded_sum = 0
    
        for expression in expressions:
            if expression not in expression_to_encoded:
                subset = merged_data[merged_data['chiefcomplaint'] == expression]
                Target_Encoder1.fit(subset['chiefcomplaint'], subset['outcome_encoded'])
                encoded_value = Target_Encoder1.transform(pd.DataFrame({'chiefcomplaint': [expression]}))['chiefcomplaint'][0]
                expression_to_encoded[expression] = encoded_value
            encoded_sum += expression_to_encoded[expression]
    
        encoded_expressions_sum.append(encoded_sum)

    merged_data['chiefcomplaint_encoded_sklearn'] = encoded_expressions_sum



# ############### Calculating for icd_title ######################
# #     encoded_expressions_icd_sum = []
# # # Dictionary to store mapping of expressions to their encoded values
# #     icd_expression_to_encoded = {}

# #     for index, row in merged_data.iterrows():
# #         expressions_icd = row['icd_title'].split(',')
# #         encoded_sum_icd = 0
    
# #         for expression in expressions_icd:
# #             if expression not in icd_expression_to_encoded:
# #                 subset = merged_data[merged_data['icd_title'] == expression]
# #                 Target_Encoder4.fit(subset['icd_title'], subset['outcome_encoded'])
# #                 encoded_icd_value = Target_Encoder4.transform(pd.DataFrame({'icd_title': [expression]}))['icd_title'][0]
# #                 icd_expression_to_encoded[expression] = encoded_icd_value
# #             encoded_sum_icd += icd_expression_to_encoded[expression]
    
# #         encoded_expressions_icd_sum.append(encoded_sum_icd)
# #     merged_data['icd_title_encoded_sklearn'] = encoded_expressions_sum

# ## **********************      End Modification           *************************

    merged_data['race_encoded_sklearn']  = Target_Encoder2.fit_transform(merged_data['race'], merged_data['outcome_encoded'])
    merged_data['arrival_transport_sklearn']  = Target_Encoder3.fit_transform(merged_data['arrival_transport'], merged_data['outcome_encoded'])

    X_train_encoded = merged_data.drop(['chiefcomplaint','race', 'arrival_transport'], axis=1)
    X_train_encoded.rename(columns={"chiefcomplaint_encoded_sklearn": "chiefcomplaint",
                                # "icd_title_encoded_sklearn": "icd_title",
                                "race_encoded_sklearn": "race",
                                "arrival_transport_sklearn": "arrival_transport"}, inplace=True)

    X_train_encoded = X_train_encoded.drop(columns=['outcome_hospitalization','outcome_encoded'])

#########################################        Working on Test

    X_test_encoded = X_test.copy()

# **********************    Start New Modification       *************************
    encoded_expressions_sum_test = []
    encoded_expressions_icd_sum_test = []

# Iterate through each row
    for index, row in X_test_encoded.iterrows():
        expressions_icd_test = row['chiefcomplaint'].split(',')
        encoded_sum_test = 0
    
        for expression in expressions_icd_test:
            if expression not in expression_to_encoded:
                nearest_indices = find_nearest_strings(list(expression_to_encoded.keys()), expression, k=5)
                nearest_values = [expression_to_encoded[list(expression_to_encoded.keys())[idx]] for idx in nearest_indices]
                mean_encoded_value = sum(nearest_values) / len(nearest_values)
                expression_to_encoded[expression] = mean_encoded_value
            encoded_sum_test += expression_to_encoded[expression]
    
        encoded_expressions_sum_test.append(encoded_sum_test)

    X_test_encoded['chiefcomplaint_encoded_sklearn'] = encoded_expressions_sum_test

# ##icd
#     for index, row in X_test_encoded.iterrows():
#         expressions_icd_test = row['icd_title'].split(',')
#         encoded_sum_icd_test = 0
    
#         for expression in expressions_icd_test:
#             if expression not in icd_expression_to_encoded:
#                 nearest_indices = find_nearest_strings(list(icd_expression_to_encoded.keys()), expression, k=5)
#                 nearest_icd_values = [icd_expression_to_encoded[list(icd_expression_to_encoded.keys())[idx]] for idx in nearest_indices]
#                 mean_icd_encoded_value = sum(nearest_icd_values) / len(nearest_icd_values)
#                 icd_expression_to_encoded[expression] = mean_icd_encoded_value
#             encoded_sum_icd_test += icd_expression_to_encoded[expression]
    
#         encoded_expressions_icd_sum_test.append(encoded_sum_icd_test)
#     X_test_encoded['icd_title_encoded_sklearn'] = encoded_expressions_icd_sum_test


# **********************      End Modification           *************************

    X_test_encoded['race_encoded_sklearn'] = Target_Encoder2.transform(X_test_encoded['race'])
    X_test_encoded['arrival_transport_sklearn'] = Target_Encoder3.transform(X_test_encoded['arrival_transport'])

    X_test_encoded = X_test_encoded.drop(['chiefcomplaint', 'race', 'arrival_transport'], axis=1)
    X_test_encoded.rename(columns={"chiefcomplaint_encoded_sklearn": "chiefcomplaint",
                                # "icd_title_encoded_sklearn": "icd_title",
                                "race_encoded_sklearn": "race",
                                "arrival_transport_sklearn": "arrival_transport"}, inplace=True)
###################################################   End Test preprocessing #################

    # X_train = update_triage_columns(X_train)
    # X_test_encoded = update_triage_columns(X_test_encoded)
    X_train_processed = process(X_train)

    X_train_scaled = X_train_processed.copy()
    # scaler.fit(X_train_processed)
    # X_train_scaled = scaler.transform(X_train_processed)

    X_test_processed = process(X_test_encoded)
    # X_test_scaled = scaler.transform(X_test_processed)
    X_test_scaled = X_test_processed.copy()


    # print("************Gradient Boosting:***********")
    # gb= GradientBoostingClassifier(random_state=random_seed)
    print("Logistic Regression:")
    from sklearn.linear_model import LogisticRegression
    gb=LogisticRegression(random_state=random_seed)


    start = time.time()
    gb.fit(X_train_scaled, y_train)
    runtime = time.time() - start
    print('Training time:', runtime, 'seconds')
    probs = gb.predict_proba(X_test_scaled)

    auc_score = roc_auc_score(y_test, probs[:, 1])
    result_list.append(auc_score)
    print('AUC:', auc_score)

    roc_auc_list, average_precision_list, sensitivity_list, specificity_list, threshold_list = PlotROCCurve(probs[:, 1],y_test, ci=confidence_interval, random_seed=random_seed)
    lower_auroc_list, upper_auroc_list, lower_ap_list, upper_ap_list, lower_sensitivity_list, upper_sensitivity_list, lower_specificity_list, upper_specificity_list = Calculatemetric_CI(probs[:, 1],y_test, ci=confidence_interval, random_seed=random_seed)

print('***** ********* Overal result ***** *********')
average_sensitivity = np.mean(sensitivity_list)
average_specificity = np.mean(specificity_list)

average_auc = np.mean(roc_auc_list)
average_precision = np.mean(average_precision_list)


lower_bound = np.percentile(lower_auroc_list, (100 - confidence_interval) / 2)
upper_bound = np.percentile(upper_auroc_list, 100 - (100 - confidence_interval) / 2)
print("Average AUC: {:.4f}".format(average_auc))
print("AUC CI ({}%): [{:.4f}, {:.4f}]".format(confidence_interval, lower_bound, upper_bound))




lower_ap = np.percentile(lower_ap_list, (100 - confidence_interval) / 2)
upper_ap = np.percentile(upper_ap_list, 100 - (100 - confidence_interval) / 2)
print("Average AUPRC: {:.4f}".format(average_precision))
print("Overall AUPRC CI ({}%): [{:.4f}, {:.4f}]".format(confidence_interval, lower_ap, upper_ap))


lower_sensitivity = np.percentile(lower_sensitivity_list, (100 - confidence_interval) / 2)
upper_sensitivity = np.percentile(upper_sensitivity_list, 100 - (100 - confidence_interval) / 2)
print("Average Sensitivity: {:.4f}".format(average_sensitivity))
print("Overall Sensitivity CI ({}%): [{:.4f}, {:.4f}]".format(confidence_interval, lower_sensitivity, upper_sensitivity))


lower_specificity = np.percentile(lower_specificity_list, (100 - confidence_interval) / 2)
upper_specificity = np.percentile(upper_specificity_list, 100 - (100 - confidence_interval) / 2)
print("Average Specificity: {:.4f}".format(average_specificity))
print("Overall Specificity CI ({}%): [{:.4f}, {:.4f}]".format(confidence_interval, lower_specificity, upper_specificity))
