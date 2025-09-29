import pandas as pd
import numpy as np

import pytest

import objective

def test_FindNeighbors():
	
	matrix = [
		[1, 5, 0],
		[2, 4, 1],
		[3, 3, 0],
		[4, 2, 1],
		[5, 1, 0]
	]
	
	columns = ['feature1', 'feature2', 'class']
	
	data_table = pd.DataFrame(matrix, columns = columns)
	
	X = data_table[['feature1', 'feature2']]
	y = data_table[['class']]
	
	i = 1
	target_neighbors, impostors = objective.FindNeighbors(i, X, y)
	
	assert target_neighbors == [3]
	assert impostors == [0, 2, 4]
	
	i = 2
	target_neighbors, impostors = objective.FindNeighbors(i, X, y)
	
	assert target_neighbors == [0, 4]
	assert impostors == [1, 3]
