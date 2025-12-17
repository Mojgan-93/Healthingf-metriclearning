from expected import *

from common import *
from joblib import Parallel, delayed


def FindNeighbors(i, X, y):
	
	assert 0 <= i and i < y.shape[0]
	
	# convert to list
	y = y.iloc[:, 0].tolist()
	
	current_y = y[i]
	
	# target neighbors & impostors
	target_neighbors = [j for j in range(len(X)) if i != j and y[j] == current_y]
	impostors        = [j for j in range(len(X)) if i != j and y[j] != current_y]
	
	return target_neighbors, impostors


def hinge_loss(x):
	
	return np.maximum(0, x)


def ObjectiveFunc_m(m, lengths_m, M_psd, gamma, lambda_, X, y):
	
	m_raw = BuildThresholdsMRaw(m, lengths_m)
	
	Matrix_E_z = GetEZMatrix(m_raw, X)
            
	matrix_var = {f'matrix_{v+1}': GetVarZMatrix(v, m_raw, X) for v in range(len(X))}

	return CalcDistance(M_psd, Matrix_E_z, matrix_var, gamma, lambda_, X, y, n_jobs=8)


def ObjectiveFunc_M(M_flattened, Matrix_E_z, matrix_var, gamma, lambda_, X, y):
	
	M = RebuildSymmetricM(M_flattened, X.shape[1])
	M = GetNearestPsd(M)

	return CalcDistance(M, Matrix_E_z, matrix_var, gamma, lambda_, X, y, n_jobs=8)


def CalcDistance(M, Matrix_E_z, matrix_var, gamma, lambda_, X, y, n_jobs=8):
    def process_point(i):
        E_distance_local = 0
        distance_cache = {}  # Local cache for each process

        # Find target and impostor neighbors
        target_neighbors, impostor_neighbors = FindNeighbors(i, X, y)
        
        for j in target_neighbors:
            if (i, j) in distance_cache:
                E_distance_ij = distance_cache[(i, j)]
            else:
                E_distance_ij = expected_distance(i, j, Matrix_E_z, 
                                                  matrix_var[f'matrix_{i+1}'], 
                                                  matrix_var[f'matrix_{j+1}'], M)
                distance_cache[(i, j)] = E_distance_ij

            E_distance_local += E_distance_ij

            for k in impostor_neighbors:
                if (i, k) in distance_cache:
                    E_distance_ik = distance_cache[(i, k)]
                else:
                    E_distance_ik = expected_distance(i, k, Matrix_E_z, 
                                                      matrix_var[f'matrix_{i+1}'], 
                                                      matrix_var[f'matrix_{k+1}'], M)
                    distance_cache[(i, k)] = E_distance_ik

                hinge_value = hinge_loss(E_distance_ij - E_distance_ik + gamma)
                E_distance_local += lambda_ * hinge_value

        return E_distance_local
    
    # **Modified for Parallelism:** Compute process_point(i) in parallel over all i
    E_distance = sum(Parallel(n_jobs=n_jobs)(delayed(process_point)(i) for i in range(len(X))))
    
    # Optionally, normalize the loss by the number of data points:
    normalized_loss = E_distance / TRAIN_SIZE
    return normalized_loss
