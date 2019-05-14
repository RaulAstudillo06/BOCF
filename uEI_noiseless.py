# Copyright (c) 2018, Raul Astudillo

import numpy as np
from GPyOpt.acquisitions.base import AcquisitionBase
from GPyOpt.core.task.cost import constant_cost_withGradients
from pathos.multiprocessing import ProcessingPool as Pool


class uEI_noiseless(AcquisitionBase):
    """
    Multi-attribute knowledge gradient acquisition function

    :param model: GPyOpt class of model.
    :param space: GPyOpt class of domain.
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer.
    :param utility: utility function. See utility class for details.
    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, utility=None):
        self.optimizer = optimizer
        self.utility = utility
        super(uEI_noiseless, self).__init__(model, space, optimizer, cost_withGradients=cost_withGradients)
        if cost_withGradients == None:
            self.cost_withGradients = constant_cost_withGradients
        else:
            print('LBC acquisition does now make sense with cost. Cost set to constant.')
            self.cost_withGradients = constant_cost_withGradients
        self.n_attributes = self.model.output_dim
        self.W_samples = np.random.normal(size=(25, self.n_attributes))
        self.n_hyps_samples = min(10, self.model.number_of_hyps_samples())
        self.use_full_support = self.utility.parameter_dist.use_full_support  # If true, the full support of the utility function distribution will be used when computing the acquisition function value.
        if self.use_full_support:
            self.utility_params_samples = self.utility.parameter_dist.support
            self.utility_prob_dist = np.atleast_1d(self.utility.parameter_dist.prob_dist)
        else:
            self.utility_params_samples = self.utility.parameter_dist.sample(10)

    def _compute_acq(self, X, parallel=True):
        """
        Computes the aquisition function

        :param X: set of points at which the acquisition function is evaluated. Should be a 2d array.
        """
        if parallel and len(X) > 1:
            marginal_acqX = self._marginal_acq_parallel(X)
        else:
            marginal_acqX = self._marginal_acq(X, self.utility_params_samples)
        #print('parallel')
        #print(marginal_acqX)
        #marginal_acqX = self._marginal_acq(X, self.utility_params_samples)
        #print('sequential')
        #print(marginal_acqX)
        if self.use_full_support:
            acqX = np.matmul(marginal_acqX, self.utility_prob_dist)
        else:
            acqX = np.sum(marginal_acqX, axis=1) / len(self.utility_params_samples)
        acqX = np.reshape(acqX, (X.shape[0], 1))
        #print(acqX)
        return acqX

    def _marginal_acq(self, X, utility_params_samples):
        """
        """
        fX_evaluated = self.model.posterior_mean_at_evaluated_points()
        n_w = self.W_samples.shape[0]
        L = len(utility_params_samples)
        marginal_acqX = np.zeros((X.shape[0], L))

        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            muX = self.model.posterior_mean(X)
            sigmaX = np.sqrt(self.model.posterior_variance(X))
            for l in range(L):
                max_valX_evaluated = np.max(self.utility.eval_func(utility_params_samples[l], fX_evaluated))
                for W in self.W_samples:
                    for i in range(X.shape[0]):
                        valx = self.utility.eval_func(utility_params_samples[l], muX[:,i] + sigmaX[:,i]*W)
                        marginal_acqX[i, l] += max(valx - max_valX_evaluated, 0)

        marginal_acqX /= (self.n_hyps_samples * n_w)
        return marginal_acqX

    def _marginal_acq_parallel(self, X):
        """
        """
        marginal_acqX = np.zeros((X.shape[0], len(self.utility_params_samples)))
        n_w = self.W_samples.shape[0]
        pool = Pool(4)
        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            pool.map(self._parallel_acq_helper, X)
            marginal_acqX += np.atleast_2d(pool.map(self._parallel_acq_helper, X))

        marginal_acqX /= (self.n_hyps_samples * n_w)
        return marginal_acqX

    def _parallel_acq_helper(self, x):
        """
        """
        x = np.atleast_2d(x)
        fX_evaluated = self.model.posterior_mean_at_evaluated_points()
        n_w = self.W_samples.shape[0]
        utility_params_samples = self.utility_params_samples
        L = len(utility_params_samples)
        marginal_acqx = np.zeros(L)
        mux = self.model.posterior_mean(x)[:,0]
        sigmax = np.sqrt(self.model.posterior_variance(x))[:,0]
        for l in range(L):
            max_valX_evaluated = np.max(self.utility.eval_func(utility_params_samples[l], fX_evaluated))
            for W in self.W_samples:
                valx = self.utility.eval_func(utility_params_samples[l], mux + sigmax * W)
                marginal_acqx[l] += max(valx - max_valX_evaluated, 0)

        return marginal_acqx

    def _compute_acq_withGradients(self, X):
        """
        """
        X = np.atleast_2d(X)
        # Compute marginal aquisition function and its gradient for every value of the utility function's parameters samples
        if self.use_full_support:
            utility_params_samples2 = self.utility.parameter_dist.support
        else:
            utility_params_samples2 = self.utility.parameter_dist.sample(1)
        marginal_acqX, marginal_dacq_dX = self._marginal_acq_with_gradient(X, utility_params_samples2)
        if self.use_full_support:
            acqX = np.matmul(marginal_acqX, self.utility_prob_dist)
            dacq_dX = np.tensordot(marginal_dacq_dX, self.utility_prob_dist, 1)
        else:
            acqX = np.sum(marginal_acqX, axis=1) / len(utility_params_samples2)
            dacq_dX = np.sum(marginal_dacq_dX, axis=2) / len(utility_params_samples2)
        acqX = np.reshape(acqX, (X.shape[0], 1))
        dacq_dX = np.reshape(dacq_dX, X.shape)
        return acqX, dacq_dX

    def _marginal_acq_with_gradient(self, X, utility_params_samples):
        """
        """
        fX_evaluated = self.model.posterior_mean_at_evaluated_points()
        X = np.atleast_2d(X)
        marginal_acqX = np.zeros((X.shape[0], len(utility_params_samples)))
        marginal_dacq_dX = np.zeros((X.shape[0], X.shape[1], len(utility_params_samples)))
        W_samples2 = self.W_samples#np.random.normal(size=(1, self.n_attributes))

        n_w = W_samples2.shape[0]
        for h in range(self.n_hyps_samples):
            self.model.set_hyperparameters(h)
            muX = self.model.posterior_mean(X)
            sigmaX = np.sqrt(self.model.posterior_variance(X))
            dmuX_dX = self.model.posterior_mean_gradient(X)
            dvar_dX = self.model.posterior_variance_gradient(X)
            for l in range(len(utility_params_samples)):
                max_valX_evaluated = np.max(self.utility.eval_func(utility_params_samples[l], fX_evaluated))
                #print(max_valX_evaluated)
                for W in W_samples2:
                    for i in range(X.shape[0]):
                        a = muX[:,i] + sigmaX[:,i]*W
                        valx = self.utility.eval_func(utility_params_samples[l], a)
                        marginal_acqX[i, l] += max(valx - max_valX_evaluated, 0)
                        if valx > max_valX_evaluated:
                            b = np.multiply((0.5 * W / sigmaX[:, i]), dvar_dX[:, i, :].transpose()).transpose()
                            b += dmuX_dX[:,i,:]
                            marginal_dacq_dX[i, :, l] += np.matmul(
                                self.utility.eval_gradient(utility_params_samples[l], a), b)

        marginal_acqX /= (self.n_hyps_samples * n_w)
        marginal_dacq_dX /= (self.n_hyps_samples * n_w)
        return marginal_acqX, marginal_dacq_dX

    def update_Z_samples(self, n_samples):
        print('Update utility parameter W and Z samples')
        self.W_samples = np.random.normal(size=self.W_samples.shape)
        #if not self.use_full_support:
            #self.utility_params_samples = self.utility.parameter_dist.sample(len(self.utility_params_samples))