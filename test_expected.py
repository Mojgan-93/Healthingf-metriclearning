import math

import pandas as pd
import numpy as np
import numpy.testing as npt

import pytest

import preprocess_car
import expected

from common import *


def test_GetThresholdsMRaw():
	
	matrix = [
		[2, 2, 1],
		[1, 3, 2],
		[2, 1, 2]
	]
	
	columns = ['feature1', 'feature2', 'feature3']
	
	feature_table = pd.DataFrame(matrix, columns = columns)
	
	m_raw = expected.GetThresholdsMRaw(feature_table)
	
	assert m_raw == [[0.5, 1.5, 2.5], [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5]]


def test_BuildThresholdsM():
	
	m_raw = [[0.5, 1.5, 2.5], [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5]]
	
	m, lengths_m = expected.BuildThresholdsM(m_raw)
	
	npt.assert_array_equal(m, np.array([0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5]))
	assert lengths_m == [3, 4, 3]


def test_BuildThresholdsMRaw():
	
	m = np.array([0.5, 1.5, 2.5, 0.5, 1.5, 2.5, 3.5, 0.5, 1.5, 2.5])
	lengths_m = [3, 4, 3]
	
	m_raw = expected.BuildThresholdsMRaw(m, lengths_m)
	
	assert m_raw == [[0.5, 1.5, 2.5], [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5]]


def test_GetEZRow():
	
	matrix = [
		[2, 2, 1],
		[1, 3, 2],
		[2, 1, 2]
	]
	
	columns = ['feature1', 'feature2', 'feature3']
	
	feature_table = pd.DataFrame(matrix, columns = columns)
	
	# the default m_raw is as following
	# m_raw = expected.GetThresholdsMRaw(feature_table)
	# m_raw = [[0.5, 1.5, 2.5], [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5]]
	
	# for testing purpose, set m_raw as:
	m_raw = [[0.7, 1.7, 2.7], [0.3, 1.3, 2.3, 3.3], [1, 2, 3]]
	
	z1 = expected.GetEZRow(1, m_raw, feature_table)
	
	assert z1 == [1.2, 2.8, 2.5]


def test_GetEZMatrix():
	
	matrix = [
		[2, 2, 1],
		[1, 3, 2],
		[2, 1, 2]
	]
	
	columns = ['feature1', 'feature2', 'feature3']
	
	feature_table = pd.DataFrame(matrix, columns = columns)
	
	m_raw = expected.GetThresholdsMRaw(feature_table)
	
	matrix_z = expected.GetEZMatrix(m_raw, feature_table)
	
	assert THRESHOLD_ONE == False
	
	# initial EZ matrix: same as original matrix
	npt.assert_array_equal(matrix_z, matrix)


def test_GetVarZMatrix():
	
	matrix = [
		[2, 2, 1],
		[1, 3, 2],
		[2, 1, 2]
	]
	
	columns = ['feature1', 'feature2', 'feature3']
	feature_table = pd.DataFrame(matrix, columns = columns)
	
	
	# with default m_raw
	
	m_raw = expected.GetThresholdsMRaw(feature_table)
	
	# [1, 3, 2]
	var_z_1 = expected.GetVarZMatrix(1, m_raw, feature_table)
	
	expected_result = np.array([
		[1 / 12, 0,      0],
		[0,      1 / 12, 0],
		[0,      0,      1 / 12]
	])
	
	npt.assert_allclose(var_z_1, expected_result)
	
	
	# with non-default m_raw
	
	m_raw = [[0.5, 1.3, 2.1], [0.8, 1.6, 2.6, 3.3], [0.4, 1.3, 2.2]]
	
	# [1, 3, 2]
	var_z_1 = expected.GetVarZMatrix(1, m_raw, feature_table)
	
	expected_result = np.array([
		[0.64 / 12, 0,         0],
		[0,         0.49 / 12, 0],
		[0,         0,         0.81 / 12]
	])
	
	npt.assert_allclose(var_z_1, expected_result)


def test_expected_distance():
	
	X , y = preprocess_car.PrepareCarData()
	
	# default m_raw
	# [[0.5, 1.5, 2.5, 3.5, 4.5], [0.5, 1.5, 2.5, 3.5, 4.5], [0.5, 1.5, 2.5, 3.5, 4.5], [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5]]
	m_raw = expected.GetThresholdsMRaw(X)
	
	Matrix_E_z = expected.GetEZMatrix(m_raw, X)	
	matrix_var = {f'matrix_{v+1}': expected.GetVarZMatrix(v, m_raw, X) for v in range(len(X))}
	M = np.eye(X.shape[1])
	
	i = 1362
	j = 160
	distance = expected.expected_distance(i, j, Matrix_E_z, matrix_var[f'matrix_{i+1}'],  matrix_var[f'matrix_{j+1}'], M)
	
	assert math.isclose(distance, 15.0)
	
	# non-defalt m_raw
	m_raw = [[0.1, 1.2, 2.0, 3.4, 4.4], [0.5, 1.5, 2.5, 3.5, 4.5], [0.5, 1.5, 2.5, 3.5, 4.5], [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5], [0.5, 1.5, 2.5, 3.5]]
	
	Matrix_E_z = expected.GetEZMatrix(m_raw, X)	
	matrix_var = {f'matrix_{v+1}': expected.GetVarZMatrix(v, m_raw, X) for v in range(len(X))}
	
	distance = expected.expected_distance(i, j, Matrix_E_z, matrix_var[f'matrix_{i+1}'],  matrix_var[f'matrix_{j+1}'], M)
	
	assert math.isclose(distance, 16.58)
