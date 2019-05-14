import numpy as np
import scipy
import GPyOpt
import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from uPI import uPI
from uEI_noiseless import uEI_noiseless
from parameter_distribution import ParameterDistribution
from utility import Utility
from expectation_utility import ExpectationUtility
import cbo
import sys
import time
if __name__ == "__main__":
    # --- Function to optimize
    d = 2 # Input dimension
    m = 5 # Number of attributes
    
    A = [[3., 5., 2., 1., 7.], [5., 2., 4., 1., 9.]]
    
    def h(X):
        X = np.atleast_2d(X)
        hX = np.zeros((m, X.shape[0]))
        for j in range(m):
            for i in range(d):
                hX[j, :] += (X[:, i] - A[i][j])**2
        #hX /= np.pi
        return hX
    
    
    # --- Objective
    objective = MultiObjective(h, as_list=False, output_dim=m)
    
    # --- Space
    space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 10), 'dimensionality': d}])
    
    # --- Model (Multi-output GP)
    n_attributes = m
    model = multi_outputGP(output_dim=n_attributes, exact_feval=[True] * m, fixed_hyps=False)
    
    # --- Initial design
    initial_design = GPyOpt.experiment_design.initial_design('random', space, 2*(d+1))
    
    # --- Parameter distribution
    parameter_support = np.ones((1,))
    parameter_dist = np.ones((1,))
    parameter_distribution = ParameterDistribution(continuous=False, support=parameter_support, prob_dist=parameter_dist)
    
    
    # --- Utility function
    c = [1., 2., 5., 2., 3.]
    def U_func(parameter, y):
        c = [1., 2., 5., 2., 3.]
        y_copy = np.squeeze(y)
        aux = np.multiply(np.exp(-y_copy/np.pi), np.cos(np.pi*y_copy))
        val = -np.dot(c, aux)
        return val
    
    
    def dU_func(parameter, y):
        c = [1., 2., 5., 2., 3.]
        y_copy = np.squeeze(y)
        m = len(y_copy)
        aux = -np.pi*np.multiply(np.exp(-y_copy/np.pi), np.sin(np.pi*y_copy)) - np.multiply(np.exp(-y_copy/np.pi), np.cos(np.pi*y_copy))/np.pi
        gradient = -np.multiply(c, aux)
        return gradient
    
    U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=False)
    
    # --- Compute real optimum value
    bounds = [(0, 10), (0,10)]
    starting_points = 10.*np.random.rand(100, 2)
    parameter = parameter_support[0]
    
    def func(x):
        x_copy = np.atleast_2d(x)
        fx = h(x_copy)
        val = U_func(parameter, fx)
        return -val
    
    best_val_found = np.inf
    
    for x0 in starting_points:
        res = scipy.optimize.fmin_l_bfgs_b(func, x0, approx_grad=True, bounds=bounds)
        if best_val_found > res[1]:
            best_val_found = res[1]
            x_opt = res[0]
    print('optimum')
    print(x_opt)
    print('optimal value')
    print(-best_val_found)
    
    # --- Acquisition optimizer
    acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='CMA', inner_optimizer='lbfgs2', space=space)
    
    # --- Acquisition function
    acquisition = uPI(model, space, optimizer=acq_opt, utility=U)
    #acquisition = uEI_noiseless(model, space, optimizer=acq_opt, utility=U)
    
    # --- Evaluator
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    
    # --- Run CBO algorithm
    
    max_iter = 50
    for i in range(1):
        filename = './experiments/test7_PI_h_noiseless_' + str(i) + '.txt'
        print(filename)
        bo_model = cbo.CBO(model, space, objective, acquisition, evaluator, initial_design)
        bo_model.run_optimization(max_iter=max_iter, parallel=False, plot=False, results_file=filename)