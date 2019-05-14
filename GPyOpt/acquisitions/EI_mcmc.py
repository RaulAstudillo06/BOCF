# Copyright (c) 2016, the GPyOpt Authors
# Licensed under the BSD 3-clause license (see LICENSE.txt)

from .EI import AcquisitionEI
from ..util.general import get_quantiles


class AcquisitionEI_MCMC(AcquisitionEI):
    """
    Integrated Expected improvement acquisition function

    :param model: GPyOpt class of model
    :param space: GPyOpt class of domain
    :param optimizer: optimizer of the acquisition. Should be a GPyOpt optimizer
    :param cost_withGradients: function
    :param jitter: positive value to make the acquisition more explorative

    .. Note:: allows to compute the Improvement per unit of cost

    """

    analytical_gradient_prediction = True

    def __init__(self, model, space, optimizer=None, cost_withGradients=None, jitter=0.01):
        super(AcquisitionEI_MCMC, self).__init__(model, space, optimizer, cost_withGradients, jitter)
        
        assert self.model.MCMC_sampler, 'Samples from the hyper-parameters are needed to compute the integrated EI'

    def _compute_acq(self,x):
        """
        Integrated Expected Improvement
        """    
        means, stds = self.model.predict(x)
        fmins = self.model.get_fmin()
        f_acqu = 0
        for m,s,fmin in zip(means, stds, fmins):
            phi, Phi, _ = get_quantiles(self.jitter, fmin, m, s)    
            f_acqu += (fmin - m + self.jitter) * Phi + s * phi
        return f_acqu/(len(means))

    def _compute_acq_withGradients(self, x):
        """
        Integrated Expected Improvement and its derivative
        """
        means, stds, dmdxs, dsdxs = self.model.predict_withGradients(x)
        fmins = self.model.get_fmin()
        f_acqu = None
        df_acqu = None
        for m, s, fmin, dmdx, dsdx in zip(means, stds, fmins, dmdxs, dsdxs):
            phi, Phi, _ = get_quantiles(self.jitter, fmin, m, s)    
            f = (fmin - m + self.jitter) * Phi + s * phi        
            df = dsdx * phi - Phi * dmdx
            if f_acqu is None:
                f_acqu = f        
                df_acqu = df
            else:
                f_acqu += f
                df_acqu += df
        return f_acqu/(len(means)), df_acqu/(len(means))
        








