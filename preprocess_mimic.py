from common import *

def map_and_replace_columns(df, columns):
    mappings = {}  
    
    for column in columns:
        unique_values = sorted(df[column].unique())
        
        mapping = {original_value: new_value for new_value, original_value in enumerate(unique_values, start=1)}
        
        df[column] = df[column].map(mapping)
        
        mappings[column] = mapping
    
    return df
def PrepareMimicData():
      
	X = pd.read_csv('/train.csv')
	y = X[['outcome_hospitalization']].rename(columns={'outcome_hospitalization': 'Class'})
  
	ordinal_columns = ['triage_acuity', 'triage_pain']
	
	
	X = X[ordinal_columns]
	X[ordinal_columns] = map_and_replace_columns(X, ordinal_columns)


	return X , y