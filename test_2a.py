import numpy as np
import scipy
import GPyOpt
import GPy
from multi_objective import MultiObjective
from multi_outputGP import multi_outputGP
from uEI_noiseless import uEI_noiseless
from uPI import uPI
from parameter_distribution import ParameterDistribution
from utility import Utility
from expectation_utility import ExpectationUtility
import cbo
import sys
import time


# --- Function to optimize
d = 3 # Input dimension
m = 4  # Number of attributes
aux_model = []
I = np.linspace(0., 1., 8)
aux_grid = np.meshgrid(I, I, I)
grid = np.array([a.flatten() for a in aux_grid]).T
kernel = GPy.kern.SE(input_dim=d, variance=2., lengthscale=0.3)
cov = kernel.K(grid)
mean = np.zeros((8 ** d,))
for j in range(m):
    r = np.random.RandomState(j+7)
    Y = r.multivariate_normal(mean, cov)
    Y = np.reshape(Y, (8 ** d, 1))
    print(Y[:5, 0])
    aux_model.append(GPy.models.GPRegression(grid, Y, kernel, noise_var=1e-10))

def h(X):
    X = np.atleast_2d(X)
    hX = np.empty((m, X.shape[0]))
    for j in range(m):
        hX[j, :] = aux_model[j].posterior_mean(X)[:, 0]
    return hX

# --- Objective
objective = MultiObjective(h, as_list=False, output_dim=m)

# --- Space
space = GPyOpt.Design_space(space=[{'name': 'var', 'type': 'continuous', 'domain': (0, 1), 'dimensionality': d}])

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
def U_func(parameter, y):
    aux = -np.exp(y)
    return np.sum(aux, axis=0)

def dU_func(parameter, y):
    return -np.exp(y)

U = Utility(func=U_func,dfunc=dU_func,parameter_dist=parameter_distribution,linear=False)

# --- Expectation of utility
def psi(parameter, mean, var):
    mean_aux = np.squeeze(mean)
    var_aux = np.squeeze(var)
    aux = mean_aux + 0.5*var_aux
    val = -np.sum(np.exp(aux))
    return val

def psi_gradient(parameter, mean, var):
    mean_aux = np.squeeze(mean)
    var_aux = np.squeeze(var)
    aux = mean_aux + 0.5*var_aux
    aux = np.exp(aux)
    gradient = -np.concatenate((aux, 0.5*aux))
    return gradient

expectation_U = ExpectationUtility(psi, psi_gradient)

# --- Compute real optimum value
bounds = [(0, 1)] * d
starting_points = np.random.rand(100, d)
opt_val = 0
parameter = parameter_support[0]

def func(x):
    x_copy = np.atleast_2d(x)
    hx = h(x_copy)
    val = U_func(parameter, hx)
    return -val


best_val_found = np.inf

for x0 in starting_points:
    res = scipy.optimize.fmin_l_bfgs_b(func, x0, approx_grad=True, bounds=bounds)
    if best_val_found > res[1]:
        best_val_found = res[1]
        x_opt = res[0]
print('optimum')
print(x_opt)
print('h(optimum)')
print(h(x_opt))
print('optimal value')
print(-best_val_found)

# --- Optimum in cluster
#optimum
#[0.56207128 0.68580576 0.84633431]
#h(optimum)
#[[-2.3841504 ]
 #[-2.98629507]
 #[-1.8404311 ]
 #[-2.05306124]]
#optimal value
#[-0.42973174]

# --- Optimum in laptop
#optimum
#optimal value
#[-0.53699102]

# --- Acquisition optimizer
acq_opt = GPyOpt.optimization.AcquisitionOptimizer(optimizer='CMA', inner_optimizer='lbfgs2', space=space)

# --- Acquisition function
#acquisition = uEI_noiseless(model, space, optimizer=acq_opt, utility=U)
acquisition = uPI(model, space, optimizer=acq_opt, utility=U)

# --- Evaluator
evaluator = GPyOpt.core.evaluators.Sequential(acquisition)

# --- Run CBO algorithm
max_iter = 50
for i in range(1):
    filename = './experiments/test5_PI_h_noiseless_' + str(i) + '.txt'
    print(filename)
    bo_model = cbo.CBO(model, space, objective, acquisition, evaluator, initial_design, expectation_utility=expectation_U)
    bo_model.run_optimization(max_iter=max_iter, parallel=False, plot=False, results_file=filename)