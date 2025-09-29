# m: 1-d array
# m_raw: list of list

import pandas as pd
import numpy as np

import scipy.linalg

#import preprocess_car
import preprocess_cancer

from common import *


def GetThresholdsMRaw(feature_table):
	
	m_raw = []
	
	for d in range(feature_table.shape[1]):
		
		unique_values = feature_table.iloc[:, d].unique()
		print(unique_values)
		min_value = unique_values.min()
		max_value = unique_values.max()
		# min_value = int(np.floor(min_value))
		# max_value = int(np.ceil(max_value))
		
		if (THRESHOLD_ONE):
			
			thresholds = [x for x in range(min_value, max_value + 2)]
			
		else:
			
			thresholds = [x - 0.5 for x in range(min_value, max_value + 2)]
		
		m_raw.append(thresholds)
		
	return m_raw


def BuildThresholdsM(m_raw):
	
	m = []
	lengths_m = []
	
	for sublist in m_raw:
		
		m.extend(sublist)
		lengths_m.append(len(sublist))
	
	return np.array(m), lengths_m


def BuildThresholdsMRaw(m, lengths_m):
	
	m_raw = []
	index = 0
	
	for length in lengths_m:
		
		m_raw.append(m[index:index + length].tolist())
		index += length
	
	return m_raw


def FlattenSymmetricM(matrix):
	n = matrix.shape[0]
	flattened = []
	for i in range(n):
		for j in range(i, n):
			flattened.append(matrix[i, j])
	return np.array(flattened)

# Rebuild the symmetric matrix from the flattened array
def RebuildSymmetricM(flattened, n):
	matrix = np.zeros((n, n))
	idx = 0
	for i in range(n):
		for j in range(i, n):
			value = flattened[idx]
			if not np.issubdtype(type(value), np.number):
				raise TypeError(f"Expected a scalar, got {type(value)}")
			matrix[i, j] = value
			if i != j:
				matrix[j, i] = value
			idx += 1
	return matrix


def GetNearestPsd(A):
	"""Return the nearest positive semi-definite matrix to A."""
	A_sym = (A + A.T) / 2
	
	eigvals, eigvecs = np.linalg.eigh(A_sym)
	
	eigvals[eigvals < 0] = 0
	
	A_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T
	
	A_psd = (A_psd + A_psd.T) / 2
	
	return A_psd


def GetEZRow(i, m_raw, feature_table):
	
	vector = []
	
	for d in range(feature_table.shape[1]):
		
		def func_md(t):
			
			md = m_raw[d]
			t_index = t - 1
			
			return md[t_index]
		
		t = feature_table.iloc[i, d]
		
		expected = func_md(t) + (0.5 * (func_md(t + 1) - func_md(t)))
		
		vector.append(expected)
	
	return vector


def GetEZMatrix(m_raw, feature_table):
	
	matrix_z = []
	
	for i in range(feature_table.shape[0]):
		
		matrix_z.append(GetEZRow(i, m_raw, feature_table))
	
	return np.array(matrix_z)


def GetEZDataFrame(m_raw, feature_table):
	
	matrix = GetEZMatrix(m_raw, feature_table)
	
	return pd.DataFrame(matrix, columns = feature_table.columns)



def GetVarZMatrix(i, m, dataset):
	
	D = dataset.shape[1]
	var_matrix = np.zeros((D, D))
	
	for d in range(D):
		
		def func_md(t):
			
			md = m[d]
			t_index = t - 1
			
			return md[t_index]
		
		t = dataset.iloc[i, d]
		
		var_d = (1/12) * (func_md(t + 1) - func_md(t)) ** 2
		
		var_matrix[d, d] = var_d
	
	return var_matrix


def expected_distance(i, j, Matrix_E_z,  matrix_var_i, matrix_var_j, M):
	
	e_z_i= Matrix_E_z[i, :]
	e_z_j= Matrix_E_z[j, :]
	e_z_i_array = np.array(e_z_i)
	e_z_j_array = np.array(e_z_j)
	
	vector = e_z_i_array - e_z_j_array
	
	transpose_vector = vector.reshape(1, -1)
	#print("Shapes:", transpose_vector.shape, M.shape, vector.shape)
	distance = transpose_vector @ M @ vector
	
	mvar_z_i = M @ matrix_var_i
	mvar_z_j = M @ matrix_var_j
	
	distance +=  np.trace(mvar_z_i)
	distance +=  np.trace(mvar_z_j)
	
	
	return distance[0]
