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
    d = 4 # Input dimension
    
    S = [0., 1., 2.5]
    T = [15., 30., 45., 60.]
    
    m = len(S)*len(T)
    
    M = 10
    D = 0.07
    L = 1.505
    tau = 30.1525
    
    def scale_domain(X):
        X_scaled = np.copy(X)
        X_scaled[:, 0] = 4.*X_scaled[:, 0] + 7.
        X_scaled[:, 1] = 0.1*X_scaled[:, 1] + 0.02
        X_scaled[:, 2] = 2.99*X_scaled[:, 2] + 0.01
        X_scaled[:, 3] = 0.285*X_scaled[:, 3] + 30.01
        return X_scaled
    
    def c(s, t, X):
        val = np.empty((X.shape[0],))
        for i in range(X.shape[0]):
            aux = X[i, 0]/np.sqrt(4*np.pi*X[i, 1]*t)*np.exp(-np.square(s)/(4*X[i, 1]*t))
            if X[i, 3] < t:
                aux += X[i, 0]/np.sqrt(4*np.pi*X[i, 1]*(t - X[i, 3]))*np.exp(-np.square(s - X[i, 2])/(4*X[i, 1]*(t - X[i, 3])))
            val[i] = aux
        val *= np.sqrt(4*np.pi)
        return val
    
    def h_unscaled(X):
        hX = np.zeros((m, X.shape[0]))
        for a in range(len(S)):
            for b in range(len(T)):
                hX[a*len(T) + b, :] = c(S[a], T[b], X)
        return hX
    
    def h(X):
        X_scaled = scale_domain(X)
        hX = np.zeros((m, X_scaled.shape[0]))
        for a in range(len(S)):
            for b in range(len(T)):
                hX[a*len(T) + b, :] = c(S[a], T[b], X_scaled)
        return hX
    
    # --- Objective
    objective = MultiObjective(h, as_list=False, output_dim=m)
    
    # --- Space
    #space = GPyOpt.Design_space(space=[{'name': 'M', 'type': 'continuous', 'domain': (7., 13.), 'dimensionality': 1}, {'name': 'D', 'type': 'continuous', 'domain': (0.02, 0.12), 'dimensionality': 1}, {'name': 'L', 'type': 'continuous', 'domain': (0.01, 3.), 'dimensionality': 1}, {'name': 'tau', 'type': 'continuous', 'domain': (30.01, 30.295), 'dimensionality': 1}])
    space = GPyOpt.Design_space(space=[{'name': 'x', 'type': 'continuous', 'domain': (0., 1.), 'dimensionality': d}])
    
    # --- Model (Multi-output GP)
    n_attributes = m
    model = multi_outputGP(output_dim=n_attributes, exact_feval=[True] * m, fixed_hyps=False)
    
    # --- Initial design
    initial_design = GPyOpt.experiment_design.initial_design('random', space, 2*(d+1))
    
    # --- Parameter distribution
    true_opt = np.reshape([M, D, L, tau], (1, 4))
    parameter_support = h_unscaled(true_opt).transpose()
    parameter_dist = np.ones((1,))
    parameter_distribution = ParameterDistribution(continuous=False, support=parameter_support, prob_dist=parameter_dist)
    
    
    # --- Utility function
    def U_func(parameter, y):
        #y_aux = np.squeeze(y)
        aux = (y.transpose() - parameter).transpose()
        return -np.sum(np.square(aux), axis=0)
    
    def dU_func(parameter, y):
        y_aux = np.squeeze(y)
        return -2*(y_aux - parameter)
    
    U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=False)
    
    # --- Expectation of utility
    def psi(parameter, mu, var):
        #mu_aux = np.squeeze(mu)
        #var_aux = np.squeeze(var)
        aux = (mu.transpose() - parameter).transpose()
        val = -np.sum(np.square(aux), axis=0) - np.sum(var, axis=0)
        return val
    
    def psi_gradient(parameter, mu, var):
        mu_aux = np.squeeze(mu)
        var_aux = np.squeeze(var)
        gradient = -np.concatenate((2*(mu_aux - parameter), np.ones((len(var_aux),))))
        return gradient
    
    expectation_U = ExpectationUtility(psi, psi_gradient)
    
    # --- Compute real optimum value
    #bounds = [(7., 13.), (0.02, 0.12), (0.01, 3.), (30.01, 30.295)]
    bounds = [(0., 1.)]*d
    starting_points = np.random.rand(100, 4)
    #starting_points[:, 0] = 4.*starting_points[:, 0] + 7.
    #starting_points[:, 1] = 0.1*starting_points[:, 1] + 0.02
    #starting_points[:, 2] = 2.99*starting_points[:, 2] + 0.01
    #starting_points[:, 3] = 0.285*starting_points[:, 3] + 30.01
    
    parameter = parameter_support[0]
    print(parameter)
    
    def func(x):
        x_copy = np.atleast_2d(x)
        fx = h(x_copy)
        val = U_func(parameter, fx)
        return -val
    
    best_val_found = np.inf
    
    for x0 in starting_points:
        res = scipy.optimize.fmin_l_bfgs_b(func, x0, approx_grad=True, bounds=bounds)
        #print(res)
        if best_val_found > res[1]:
            best_val_found = res[1]
            x_opt = res[0]
    print('optimum')
    print(x_opt)
    print('best value found')
    print(-best_val_found)
    print('true optimum')
    print(0.)
    
    
    # --- Acquisition optimizer
    acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='CMA', inner_optimizer='lbfgs2', space=space)
    
    # --- Acquisition function
    acquisition = uPI(model, space, optimizer=acq_opt, utility=U)
    
    # --- Evaluator
    evaluator = GPyOpt.core.evaluators.Sequential(acquisition)
    
    # --- Run CBO algorithm
    
    max_iter = 3
    for i in range(1):
        filename = './experiments_local/test8_PI_h_noiseless_' + str(i) + '.txt'
        bo_model = cbo.CBO(model, space, objective, acquisition, evaluator, initial_design, expectation_utility=expectation_U)
        bo_model.run_optimization(max_iter=max_iter, parallel=False, plot=False, results_file=filename)