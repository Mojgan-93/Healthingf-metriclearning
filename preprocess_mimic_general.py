from common import *
from sklearn.preprocessing import LabelEncoder
from category_encoders import TargetEncoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler

encoder = LabelEncoder()
train = pd.read_csv('/train.csv')
test = pd.read_csv('/test.csv')

variable = ["age", "gender", 
            
            "n_ed_30d", "n_ed_90d", "n_ed_365d", "n_hosp_30d", "n_hosp_90d", 
            "n_hosp_365d", "n_icu_30d", "n_icu_90d", "n_icu_365d", 
            
            "triage_temperature", "triage_heartrate", "triage_resprate", 
            "triage_o2sat", "triage_sbp", "triage_dbp", "triage_pain", "triage_acuity",
            
            "chiefcom_chest_pain", "chiefcom_abdominal_pain", "chiefcom_headache",
            "chiefcom_shortness_of_breath", "chiefcom_back_pain", "chiefcom_cough", 
            "chiefcom_nausea_vomiting", "chiefcom_fever_chills", "chiefcom_syncope", 
            "chiefcom_dizziness", 
            
            "cci_MI", "cci_CHF", "cci_PVD", "cci_Stroke", "cci_Dementia", 
            "cci_Pulmonary", "cci_Rheumatic", "cci_PUD", "cci_Liver1", "cci_DM1", 
            "cci_DM2", "cci_Paralysis", "cci_Renal", "cci_Cancer1", "cci_Liver2", 
            "cci_Cancer2", "cci_HIV", 
            
            "eci_Arrhythmia", "eci_Valvular", "eci_PHTN",  "eci_HTN1", "eci_HTN2", 
            "eci_NeuroOther", "eci_Hypothyroid", "eci_Lymphoma", "eci_Coagulopathy", 
            "eci_Obesity", "eci_WeightLoss", "eci_FluidsLytes", "eci_BloodLoss",
            "eci_Anemia", "eci_Alcohol", "eci_Drugs","eci_Psychoses", "eci_Depression",
            
            "race", "arrival_transport", "chiefcomplaint"]

outcome = "outcome_hospitalization"

def map_and_replace_columns(df, columns):
    mappings = {}
    
    for column in columns:
        values = df[column].values
        rounded_values = np.ceil(values).astype(int)
        unique_values = np.unique(rounded_values)
        
        mapping = {old_value: new_value for new_value, old_value in enumerate(unique_values, start=1)}
        mapped_values = np.array([mapping[value] for value in rounded_values])
        
        df[column] = mapped_values
        mappings[column] = mapping  
    
    return df, mappings


def PrepareMimicData():
	
	ordinal_columns = ['triage_acuity', 'triage_pain']
	
	
	train = train[ordinal_columns]
	train[ordinal_columns] = map_and_replace_columns(train, ordinal_columns)

	y = train['outcome_hospitalization']

	
	return train , y
	
def chief_preprocess(data):
    data.loc[:, 'chiefcomplaint'] = data['chiefcomplaint'].str.lower()
    data['chiefcomplaint'].fillna('notreported', inplace=True)
 
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

def preprocess_mimic_data(X_train, X_test, y_train, y_test):
    Target_Encoder1 = TargetEncoder()
    Target_Encoder2 = TargetEncoder()
    Target_Encoder3 = TargetEncoder()
    Target_Encoder4 = TargetEncoder()

    X_train = chief_preprocess(X_train)
    X_test  = chief_preprocess(X_test)
   
    X_train.loc[:, 'gender'] = encoder.fit_transform(X_train['gender'])
    X_test.loc[:, 'gender'] = encoder.transform(X_test['gender'])

    # X_train['ed_los'] = pd.to_timedelta(X_train['ed_los']).dt.seconds / 60
    # X_test['ed_los'] = pd.to_timedelta(X_test['ed_los']).dt.seconds / 60

    merged_data = pd.merge(X_train, y_train,  left_index=True, right_index=True)

    merged_data['outcome_encoded'] = encoder.fit_transform(merged_data['outcome_hospitalization'])
    ## ***** preprocess train *****

    ## **********************    Start New Modification       *************************
    encoded_expressions_sum = []
    expression_to_encoded = {}
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

    merged_data['race_encoded_sklearn']  = Target_Encoder2.fit_transform(merged_data['race'], merged_data['outcome_encoded'])
    merged_data['arrival_transport_sklearn']  = Target_Encoder3.fit_transform(merged_data['arrival_transport'], merged_data['outcome_encoded'])
    X_train_encoded = merged_data.drop(['chiefcomplaint', 'race', 'arrival_transport'], axis=1)
    X_train_encoded.rename(columns={"chiefcomplaint_encoded_sklearn": "chiefcomplaint",
                                "race_encoded_sklearn": "race",
                                "arrival_transport_sklearn": "arrival_transport"}, inplace=True)
    X_train_encoded = X_train_encoded.drop(columns=['outcome_hospitalization','outcome_encoded'])


    X_test_encoded = X_test.copy()

    encoded_expressions_sum_test = []
    encoded_expressions_icd_sum_test = []
    for index, row in X_test_encoded.iterrows():
        expressions_test = row['chiefcomplaint'].split(',')
        encoded_sum_test = 0
    
        for expression in expressions_test:
            if expression not in expression_to_encoded:
                nearest_indices = find_nearest_strings(list(expression_to_encoded.keys()), expression, k=5)
                nearest_values = [expression_to_encoded[list(expression_to_encoded.keys())[idx]] for idx in nearest_indices]
                mean_encoded_value = sum(nearest_values) / len(nearest_values)
                expression_to_encoded[expression] = mean_encoded_value
            encoded_sum_test += expression_to_encoded[expression]
    
        encoded_expressions_sum_test.append(encoded_sum_test)
    X_test_encoded['chiefcomplaint_encoded_sklearn'] = encoded_expressions_sum_test

    

    X_test_encoded['race_encoded_sklearn'] = Target_Encoder2.transform(X_test_encoded['race'])
    X_test_encoded['arrival_transport_sklearn'] = Target_Encoder3.transform(X_test_encoded['arrival_transport'])
    X_test_encoded.rename(columns={"chiefcomplaint_encoded_sklearn": "chiefcomplaint",
                                # "icd_title_encoded_sklearn": "icd_title",
                                "race_encoded_sklearn": "race",
                                "arrival_transport_sklearn": "arrival_transport"}, inplace=True)

    X_train_processed = process(X_train_encoded)
    X_test_processed = process(X_test_encoded)


    return X_train_processed, X_test_processed






