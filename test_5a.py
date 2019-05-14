import numpy as np
import scipy
import GPyOpt
import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from maEI import maEI
from uEI_noiseless import uEI_noiseless
from parameter_distribution import ParameterDistribution
from utility import Utility
from expectation_utility import ExpectationUtility
import cbo
import sys
import time

# --- Function to optimize
d = 5
m = 2*(d - 1)

def h(X):
    X = np.atleast_2d(X)
    hX = np.empty((m, X.shape[0]))
    for j in range(d-1):
        hX[j,:] = X[:, j]
        hX[j+d-1,:] = X[:, j+1] - X[:, j]**2
    return hX

#noise_var = [0.25]*m
objective = MultiObjective(h, as_list=False, output_dim=m)

# --- Space
space = GPyOpt.Design_space(space =[{'name': 'var', 'type': 'continuous', 'domain': (-2,2), 'dimensionality': d}])

# --- Model (Multi-output GP)
n_attributes = m
model = multi_outputGP(output_dim=n_attributes, exact_feval=[True]*m, fixed_hyps=False)
#model = multi_outputGP(output_dim=n_attributes, noise_var=noise_var, fixed_hyps=True)

# --- Initial design
initial_design = GPyOpt.experiment_design.initial_design('random', space, 2*(d+1))
# --- Parameter distribution
parameter_support = np.atleast_1d([1.])
parameter_dist = np.ones((1,))
parameter_distribution = ParameterDistribution(continuous=False, support=parameter_support, prob_dist=parameter_dist)
#parameter_distribution = ParameterDistribution(continuous=True, sample_generator=beta_sampler)

# --- Utility function
def U_func(a, y):
    val = 0
    for j in range(d-1):
        val -= (a - y[j])**2 + 100*y[j+d-1]**2
    return val

def dU_func(a, y):
    gradient = np.empty((m, ))
    for j in range(d-1):
        gradient[j] = 2*(a - y[j])
        gradient[j+d-1] = -200*y[j+d-1]
    return gradient

U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=False)

# --- Expectation of utility
def psi(a, mean, var):
    val = 0
    for j in range(d-1):
        val -= (a - mean[j])**2 + 100*mean[j+d-1]**2 + var[j] + 100*var[j+d-1]
    return val

def psi_gradient(a, mean, var):
    gradient = np.empty((2*m, ))
    for j in range(d-1):
        gradient[j] = 2*(a - mean[j])
        gradient[j+d-1] = -200*mean[j+d-1]
        gradient[j + 2*(d-1)] = -1.
        gradient[j + 3*(d-1)] = -100.
    return gradient

expectation_U = ExpectationUtility(psi, psi_gradient)

# --- Compute real optimum value
if True:
    bounds = [(-2, 2)]*d
    starting_points = 4.*np.random.rand(100, d) - 2.
    parameter = parameter_support[0]

    def func(x):
        x_copy = np.atleast_2d(x)
        fx = h(x_copy)
        val = U_func(parameter, fx)
        return -val


    best_val_found = np.inf

    for x0 in starting_points:
        res = scipy.optimize.fmin_l_bfgs_b(func, x0, approx_grad=True, bounds=bounds)
        # print(res)
        if best_val_found > res[1]:
            best_val_found = res[1]
            x_opt = res[0]
    print('optimum')
    print(x_opt)
    print('best value found')
    print(-best_val_found)
    print('true optimum')
    print(0.)

# --- Optimum in cluster
#

# --- Acquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='lbfgs2', inner_optimizer='lbfgs2', space=space)

# --- Aquisition function
acquisition = uEI_noiseless(model, space, optimizer=acq_opt, utility=U)

# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

# --- Run optimization
max_iter = 50
for i in range(1):
    filename = './experiments_local/test9_EIh_noiseless_' + str(i) + '.txt'
    bo_model = cbo.CBO(model, space, objective, acquisition, evaluator, initial_design,
                       expectation_utility=expectation_U)
    bo_model.run_optimization(max_iter=max_iter, parallel=False, plot=False, results_file=filename)