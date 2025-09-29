from scipy.optimize import minimize
import concurrent.futures

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from expected import *

import preprocess_car

import preprocess_cancer

import objective

from common import *

from scipy.interpolate import PchipInterpolator

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="joblib")

# Variables
iteration_counter = 0 

def save_convergence_plot(values, filename, title):

    iterations, obj_values = zip(*values)
    
    # Convert data to NumPy arrays
    iterations = np.array(iterations, dtype=int)
    obj_values = np.array(obj_values, dtype=float)

    # Use Piecewise Cubic Hermite Interpolation (PCHIP) for smooth yet monotonic interpolation
    if len(iterations) > 2:
        smooth_x = np.linspace(min(iterations), max(iterations), 300)
        interpolator = PchipInterpolator(iterations, obj_values)
        smooth_y = interpolator(smooth_x)
    else:
        smooth_x, smooth_y = iterations, obj_values  # No smoothing if too few points

    plt.ioff()
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(smooth_x, smooth_y, linestyle='-', linewidth=2, color='blue')
    ax.scatter(iterations, obj_values, color='blue', s=10, edgecolors='blue')

    ax.set_xlabel("Iteration", fontsize=12, fontname="Times New Roman")
    ax.set_ylabel("Objective Function Value", fontsize=12, fontname="Times New Roman")
    ax.set_title(title, fontsize=14, fontname="Times New Roman")

    # x-axis ticks only at the first and last points
    ax.set_xticks([iterations[0], iterations[-1]])
    ax.set_xticklabels([str(iterations[0]), str(iterations[-1])], fontsize=10, fontname="Times New Roman")

    ax.tick_params(axis='both', labelsize=10)

    # **Keep the full frame** by ensuring all spines are visible
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    # Save with high resolution
    plt.savefig(filename, dpi=600, bbox_inches="tight")
    plt.close()

    print(f"Convergence plot saved as {filename}")

def CreateConstraints(lengths_m):
	
	cons = []
	
	start_idx = 0
	
	for length in lengths_m:
		
		cons.append({'type': 'ineq', 'fun': lambda x, i = start_idx: x[i]})
		
		for i in range(length - 1):
			
			index = start_idx + i
			
			cons.append({'type': 'ineq', 'fun': lambda x, i = index: x[i + 1] - x[i] - SMALL_DIFFERENCE})
		
		start_idx += length
	
	return cons


def spd_constraint(flattened):
	
	M = RebuildSymmetricM(flattened, 2)
	eigenvalues = np.linalg.eigvalsh(M)
	return np.min(eigenvalues)

def GetOptimizedM(X, y):
    start_time = time.time()
    D = X.shape[1]
    
    # Hyper parameters
    gamma = 1
    lambda_ = 1
    
    # Other parameters
    m_raw = GetThresholdsMRaw(X)
    m, lengths_m = BuildThresholdsM(m_raw)
    
    M = np.eye(D)
    M_psd = GetNearestPsd(M)
    
    # Define the optimization functions for m and M
    def optimize_m():

       #global iteration_counter1

        iteration_counter1 = 0

        initial1_obj_val = []

        cons_m = CreateConstraints(lengths_m)

        def callback(m_current):

            nonlocal iteration_counter1

            iteration_counter1 += 1  
        
            obj_val_m1 = objective.ObjectiveFunc_m(m_current, lengths_m, M_psd, gamma, lambda_, X, y)
            
            initial1_obj_val.append((iteration_counter1, obj_val_m1))

            with open(f"plots/convergence_results_firstop_{TRAIN_SIZE}.dat", "a") as file:
                file.write(f"{iteration_counter1} {obj_val_m1}\n")

        options = {'disp': True, 'maxiter': 500, 'ftol': 1e-5, 'eps': 1e-5}

        result_m = minimize(objective.ObjectiveFunc_m, m, args=(lengths_m, M_psd, gamma, lambda_, X, y), method='SLSQP', constraints=cons_m, options=options, callback=callback)
        
        m_optimized = result_m.x
        
        m_optimized_raw = BuildThresholdsMRaw(m_optimized, lengths_m)
        
        PrintEndline()
        PrintDoingTime(start_time, 'optimizing m')
        PrintEndline()

         # Ensure at least one value is stored to avoid errors
        if len(initial1_obj_val) == 1:  
            final_obj_val = result_m.fun  # Get the last known function value
            initial1_obj_val.append((iteration_counter1 + 1, final_obj_val))

        save_convergence_plot(initial1_obj_val, f"plots/first_m_optimization_convergence_{TRAIN_SIZE}.png", "Convergence of First m Optimization")

        return m_optimized, m_optimized_raw

    def optimize_M(m_optimized_raw):

        global iteration_counter

        iteration_counter = 0
        
        objective_values_M = [] 

        M_flattened = FlattenSymmetricM(M)

        Matrix_E_z = GetEZMatrix(m_optimized_raw, X)

        matrix_var = {f'matrix_{v+1}': GetVarZMatrix(v, m_optimized_raw, X) for v in range(len(X))}

        initial_obj_val = objective.ObjectiveFunc_M(M_flattened, Matrix_E_z, matrix_var, gamma, lambda_, X, y)

        objective_values_M.append((0, initial_obj_val))

        def callback1(M_flattened):

            """Callback function to track optimization process at each iteration."""
            global iteration_counter

            iteration_counter += 1
            
            obj_val = objective.ObjectiveFunc_M(M_flattened, Matrix_E_z, matrix_var, gamma, lambda_, X, y)

            objective_values_M.append((iteration_counter, obj_val))

            with open(f"plots/convergence_M_secondopt_{TRAIN_SIZE}.dat", "a") as file:
                file.write(f"{iteration_counter} {obj_val}\n")

        cons_M = {'type': 'ineq', 'fun': spd_constraint}

        options = {'disp': True, 'maxiter': 500, 'ftol': 1e-5, 'eps': 1e-5}

        result_M = minimize(objective.ObjectiveFunc_M, M_flattened, args=(Matrix_E_z, matrix_var, gamma, lambda_, X, y), method='SLSQP', constraints=cons_M, options=options, callback=callback1)

        M_optimized_flatten = result_M.x

        M_optimized = RebuildSymmetricM(M_optimized_flatten, D)

        M_optimized_psd = GetNearestPsd(M_optimized)

        PrintEndline()

        PrintDoingTime(start_time, 'optimizing M')
        
        save_convergence_plot(objective_values_M, f"plots/M_optimization_convergence_{TRAIN_SIZE}.png", "Convergence of M Optimization")

        print("Success:", result_M.success)
        print("Status:", result_M.status)
        print("Message:", result_M.message)
        print("Number of iterations:", result_M.nit)

        return M_optimized, M_optimized_flatten, M_optimized_psd, Matrix_E_z, matrix_var

    # Run optimizations in parallel using ThreadPoolExecutor
    with concurrent.futures.ThreadPoolExecutor() as executor:

        future_m = executor.submit(optimize_m)

        m_optimized, m_optimized_raw = future_m.result()

        future_M = executor.submit(optimize_M, m_optimized_raw)
        
        M_optimized, M_optimized_flatten, M_optimized_psd, Matrix_E_z, matrix_var = future_M.result()

	# 3 Again optimize m, base on the optimised M and start m with m_optimized
    m_optimized, lengths_m2 = BuildThresholdsM(m_optimized_raw)
	
    cons_m = CreateConstraints(lengths_m2)

    initial2_obj_val2 = []

    global iteration_counter2

    iteration_counter2 = 0

    def callback2(m_current):

        global iteration_counter2
        
        iteration_counter2 += 1 

        obj_val_m2 = objective.ObjectiveFunc_m(m_current, lengths_m2, M_optimized_psd, gamma, lambda_, X, y)

        initial2_obj_val2.append((iteration_counter2, obj_val_m2))

        with open(f"plots/convergence_Final_{TRAIN_SIZE}.dat", "a") as file:
                file.write(f"{iteration_counter2} {obj_val_m2}\n")
        
    options = {'disp': True, 'maxiter': 500, 'ftol': 1e-5, 'eps': 1e-3}
    result_m = minimize(objective.ObjectiveFunc_m, m_optimized, args = (lengths_m2, M_optimized_psd, gamma, lambda_, X, y), method = 'SLSQP', constraints = cons_m, options=options, callback=callback2)

    m_optimized = result_m.x

    m_optimized_raw = BuildThresholdsMRaw(m_optimized, lengths_m2)
	
    PrintEndline()
    start_time = PrintDoingTime(start_time, 'second optimizing m')
    PrintEndline()

    save_convergence_plot(initial2_obj_val2, f"plots/Last_m_optimization_convergence_{TRAIN_SIZE}.png", "Convergence of Second/Last m Optimization")

    # 4 get general result
	
    objective_value_initial  = objective.ObjectiveFunc_m(m,                   lengths_m,  M_psd,      gamma, lambda_, X, y)
    objective_value_intermed = objective.ObjectiveFunc_m(m_optimized,         lengths_m,  M_psd,      gamma, lambda_, X, y)
    objective_value_final    = objective.ObjectiveFunc_M(M_optimized_flatten, Matrix_E_z, matrix_var, gamma, lambda_, X, y)
	
    result = {
		'gamma': gamma,
		'lambda_': lambda_,
		'm_raw': m_raw,
		'm': m,
		'lengths_m': lengths_m,
		'm_optimized': m_optimized,
		'm_optimized_raw': m_optimized_raw,
		'M': M,
		'M_psd': M_psd,
		'M_optimized': M_optimized,
		'M_optimized_flatten': M_optimized_flatten,
		'Matrix_E_z': Matrix_E_z,
		'matrix_var': matrix_var,
		'objective_value_initial': objective_value_initial,
		'objective_value_intermed': objective_value_intermed,
		'objective_value_final': objective_value_final,
		'M_optimized_psd': M_optimized_psd}
	
    filename = "M_optimized_matrix.txt"
    np.savetxt(filename, M_optimized, fmt='%.6f', delimiter=' ', newline='\n')


    PrintEndline()
    start_time = PrintDoingTime(start_time, 'generating results')
	
    return result


def OptimizeRun():
	
	#X , y = preprocess_car.PrepareCarData() 
	X , y = preprocess_cancer.PrepareCarData()
	
	X = X.sample(n = 10, random_state = 42)
	y = y.sample(n = 10, random_state = 42)
	
	m_raw = GetThresholdsMRaw(X)
	print("Initial m raw:", m_raw)
	
	result = GetOptimizedM(X, y)
	print("Optimized m raw:", result['m_optimized_raw'])


if __name__ == '__main__':
	
	OptimizeRun()
