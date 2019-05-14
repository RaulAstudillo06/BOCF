import numpy as np
import scipy
import GPyOpt
import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from maEI import maEI
from maPI import maPI
from parameter_distribution import ParameterDistribution
from utility import Utility
from expectation_utility import ExpectationUtility
import cbo
import sys
import time
if __name__ == "__main__":
    # --- Function to optimize
    d = 5
    
    def f(X):
        X = np.atleast_2d(X)
        fX = np.zeros((X.shape[0], 1))
        for j in range(d-1):
            fX [:, 0] -= 100*(X[:, j+1] - X[:, j]**2)**2 + (1 - X[:, j])**2
        return fX

    # --- Objective
    objective = MultiObjective([f], as_list=True, output_dim=1)
    
    # --- Space
    space = GPyOpt.Design_space(space=[{'name': 'var1', 'type': 'continuous', 'domain': (-2, 2), 'dimensionality': d}])
    
    # --- Model (Multi-output GP)
    model = multi_outputGP(output_dim=1, exact_feval=[True], fixed_hyps=False)
    
    # --- Initial design
    initial_design = GPyOpt.experiment_design.initial_design('random', space, 2*(d+1))
    
    # --- Parameter distribution
    parameter_support = np.ones((1,))
    parameter_dist = np.ones((1,))
    parameter_distribution = ParameterDistribution(continuous=False, support=parameter_support, prob_dist=parameter_dist)
    
    
    # --- Utility function
    # --- Parameter distribution
    parameter_support = np.ones((1,1))
    parameter_dist = np.ones((1,))
    parameter_distribution = ParameterDistribution(continuous=False, support=parameter_support, prob_dist=parameter_dist)
    
    # --- Utility function
    def U_func(parameter,y):
        return np.dot(parameter,y)
    
    def dU_func(parameter,y):
        return parameter
    
    U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=True)
    
    # --- Compute real optimum value
    bounds = [(-2, 2)]*d
    starting_points = 4.*np.random.rand(100, d) -2.
    parameter = parameter_support[0]
    
    def func(x):
        x_copy = np.atleast_2d(x)
        fx = f(x_copy)
        return -fx
    
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
    acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs2', inner_optimizer='lbfgs2', space=space)
    
    # --- Acquisition function
    acquisition = maPI(model, space, optimizer=acq_opt, utility=U)
    
    # --- Evaluator
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    
    # --- Run CBO algorithm
    
    max_iter = 4
    for i in range(1):
        filename = './experiments_local/test9_PI_f_noiseless_' + str(i) + '.txt'
        print(filename)
        bo_model = cbo.CBO(model, space, objective, acquisition, evaluator, initial_design)
        bo_model.run_optimization(max_iter=max_iter, parallel=False, plot=False, results_file=filename)