"""
fitting functions for orientation correlation functions
"""

import numpy as np
from scipy.optimize import curve_fit

__all__ = ['ed_fitting_function', 'ee_fitting_function']
__author__ = ['Duncan Campbell']


class ee_fitting_function(object):
    """
    fitting function class for elipticity-elipticity (EE) correlation functions
    """
    def __init__(self):
        """
        """
        self.set_default_params()
        self.set_param_bounds()

    def set_default_params(self):
        """
        """
        d = {'A1': 0.5,
             'B1': 0.2,
             'gamma1': 0.8,
             'A2': 0.0005,
             'B2': 20.0,
             'alpha': -1.5,
             'beta': -2.3,
             'k': 0.1,
             'B3': 2.0,
             'gamma2': 1.5}
        self.params = d

    def set_param_bounds(self):
        """
        """
        d = {'A1': [0, np.inf],
             'B1': [0.0, np.inf],
             'gamma1': [-10, 10],
             'A2': [0, np.inf],
             'B2': [10.0, np.inf],
             'alpha': [-10, 0],
             'beta': [-10, 0],
             'k': [0.05, 0.3],
             'B3': [1.0, np.inf],
             'gamma2': [1.0, 10.0]}
        self.param_bounds = d

        # check if parameters are within bounds
        for key in self.params.keys():
            msg = '{0} parameter not within bounds'.format(key)
            assert (self.params[key] >= self.param_bounds[key][0]), msg
            assert (self.params[key] <= self.param_bounds[key][1]), msg

    def return_bounds_list(self, keys=None):

        if keys is None:
            keys = ['A1', 'B1', 'gamma1',
                    'A2', 'B2', 'alpha', 'beta', 'k',
                    'B3', 'gamma2']

        bounds = [[], []]
        for key in keys:
            bounds[0].append(self.param_bounds[key][0])
            bounds[1].append(self.param_bounds[key][1])
        return bounds

    def return_param_list(self, keys=None):

        if keys is None:
            keys = ['A1', 'B1', 'gamma1',
                    'A2', 'B2', 'alpha', 'beta', 'k',
                    'B3', 'gamma2']

        params = []
        for key in keys:
            params.append(self.params[key])
        return params

    def update_params(self, params):
        """
        """
        for key in params:
            try:
                self.params[key] = params[key]
            except KeyError:
                print('warning: {0} key not in paramater dictionary.'.format(key))

        # check if parameters are within bounds
        for key in self.params.keys():
            msg = '{0} parameter not within bounds'.format(key)
            assert (self.params[key] >= self.param_bounds[key][0]), msg
            assert (self.params[key] <= self.param_bounds[key][1]), msg

    def f1(self, r, params=None):
        """
        1-halo term
        """
        if params is not None:
            self.update_params(params)

        A = self.params['A1']
        B = self.params['B1']
        gamma = self.params['gamma1']
        return A*np.exp(-1.0*(r/B)**gamma)

    def f2(self, r, params=None):
        """
        two halo term
        """
        if params is not None:
            self.update_params(params)

        A = self.params['A2']
        B = self.params['B2']
        alpha = self.params['alpha']
        beta = self.params['beta']
        k = self.params['k']
        B0 = self.params['B3']
        gamma = self.params['gamma2']
        pl_args = [A, B, alpha, beta, k]  # power law args
        return _smooth_broken_powerlaw(r, *pl_args)*np.exp(-1.0*(B0/r)**gamma)

    def p1(self, r, params=None):
        """
        power law 1
        """
        if params is not None:
            self.update_params(params)

        A = self.params['A2']
        B = self.params['B2']
        alpha = self.params['alpha']
        return A*(r/B)**alpha

    def p2(self, r, params=None):
        """
        power law 2
        """
        if params is not None:
            self.update_params(params)

        A = self.params['A2']
        B = self.params['B2']
        beta = self.params['beta']
        return A*(r/B)**beta

    def fitting_function(self, r, params=None):
        """
        """
        if params is not None:
            self.update_params(params)

        return self.f1(r) + self.f2(r)

    def fit(self, x, y, yerr, params_to_fit=None, p0=None):
        """
        """

        # set default behaviour
        if params_to_fit is None:
            params_to_fit = ['A1', 'B1', 'gamma1',
                             'A2', 'B2', 'alpha', 'beta', 'k',
                             'B3', 'gamma2']
        if p0 is None:
            p0 = []
            for key in params_to_fit:
                p0.append(self.params[key])

        d = {}
        for i, key in enumerate(params_to_fit):
            d[key] = p0[i]

        def f(r, *args):
            # update paramaters
            for key in params_to_fit:
                for i, key in enumerate(params_to_fit):
                    d[key] = args[i]
            return self.fitting_function(r, d)

        bounds = self.return_bounds_list(params_to_fit)
        popt, pcov = curve_fit(f, x, y, p0=p0, sigma=yerr, bounds=bounds, maxfev=10000)

        d = {}
        for i, key in enumerate(params_to_fit):
            d[key] = popt[i]

        self.update_params(d)

        return popt, pcov


class ed_fitting_function(object):
    """
    fitting function class for elipticity-direction (ED) correlation functions
    """
    def __init__(self):
        """
        """
        self.set_default_params()
        self.set_param_bounds()

    def set_default_params(self):
        """
        """
        d = {'A1': 0.5,
             'B1': 0.33,
             'gamma1': 1.1,
             'A2': 0.1,
             'B2': 1.3,
             'gamma2': 1.6,
             'A3': 0.008,
             'B3': 23.0,
             'alpha': -1.25,
             'beta': -1.85,
             'k': 0.1,
             'B4': 5.5,
             'gamma3': 0.6}
        self.params = d

    def set_param_bounds(self):
        """
        """
        d = {'A1': [0, np.inf],
             'B1': [0, np.inf],
             'gamma1': [-10, 10],
             'A2': [0, np.inf],
             'B2': [0, np.inf],
             'gamma2': [-10, 10],
             'A3': [0, np.inf],
             'B3': [0, np.inf],
             'alpha': [-10, 0],
             'beta': [-10, 0],
             'k': [0, 1],
             'B4': [0, np.inf],
             'gamma3': [-10, 10]}
        self.param_bounds = d

        # check if parameters are within bounds
        for key in self.params.keys():
            msg = '{0} parameter not within bounds'.format(key)
            assert (self.params[key] >= self.param_bounds[key][0]), msg
            assert (self.params[key] <= self.param_bounds[key][1]), msg

    def return_bounds_list(self, keys=None):

        if keys is None:
            keys = ['A1', 'B1', 'gamma1',
                    'A2', 'B2', 'gamma2',
                    'A3', 'B3', 'alpha', 'beta', 'k',
                    'B4', 'gamma3']

        bounds = [[], []]
        for key in keys:
            bounds[0].append(self.param_bounds[key][0])
            bounds[1].append(self.param_bounds[key][1])
        return bounds

    def return_param_list(self, keys=None):

        if keys is None:
            keys = ['A1', 'B1', 'gamma1',
                    'A2', 'B2', 'gamma2',
                    'A3', 'B3', 'alpha', 'beta', 'k',
                    'B4', 'gamma3']

        params = []
        for key in keys:
            params.append(self.params[key])
        return params

    def update_params(self, params):
        """
        """
        for key in params:
            try:
                self.params[key] = params[key]
            except KeyError:
                print('warning: {0} key not in paramater dictionary.'.format(key))

        # check if parameters are within bounds
        for key in self.params.keys():
            msg = key + '{0} parameter not within bounds'.format(key)
            assert (self.params[key] >= self.param_bounds[key][0]), msg
            assert (self.params[key] <= self.param_bounds[key][1]), msg

    def f1(self, r, params=None):
        """
        cen-sat + sat-cen one-halo term
        """
        if params is not None:
            self.update_params(params)

        A = self.params['A1']
        B = self.params['B1']
        gamma = self.params['gamma1']
        return A*np.exp(-1.0*(r/B)**gamma)

    def f2(self, r, params=None):
        """
        sat-sat one-halo term
        """
        if params is not None:
            self.update_params(params)

        A = self.params['A2']
        B = self.params['B2']
        gamma = self.params['gamma2']
        return A*1.0/(1.0+np.exp((r/B)**gamma))

    def f3(self, r, params=None):
        """
        two halo term
        """
        if params is not None:
            self.update_params(params)

        A = self.params['A3']
        B = self.params['B3']
        alpha = self.params['alpha']
        beta = self.params['beta']
        k = self.params['k']
        B0 = self.params['B4']
        gamma = self.params['gamma3']
        pl_args = [A, B, alpha, beta, k]  # power law args
        return _smooth_broken_powerlaw(r, *pl_args)*np.exp(-1.0*(B0/r)**gamma)
        #return _smooth_broken_powerlaw(r, *pl_args)*1.0/np.cosh((B0/r)**gamma)
        #return _smooth_broken_powerlaw(r, *pl_args)*(1.0-1.0/np.cosh((r/B0)**gamma))

    def p1(self, r, params=None):
        """
        power law 1
        """
        if params is not None:
            self.update_params(params)

        A = self.params['A3']
        B = self.params['B3']
        alpha = self.params['alpha']
        return A*(r/B)**alpha

    def p2(self, r, params=None):
        """
        power law 2
        """
        if params is not None:
            self.update_params(params)

        A = self.params['A3']
        B = self.params['B3']
        beta = self.params['beta']
        return A*(r/B)**beta

    def fitting_function(self, r, params=None):
        """
        """
        if params is not None:
            self.update_params(params)

        return self.f1(r) + self.f2(r) + self.f3(r)


def _smooth_broken_powerlaw(x, A=1.0, x0=1, alpha=0, beta=-4, k=0.1):
    """
    smoothly transitioning double powerlaw function

    Parameters
    ----------
    A : float
        normalization constant

    x0 : float
        normalization and transition scale

    alpha : float
        x < x0 power law exponent

    beta : float
        x > x0 power law exponent

    k : float
        smoothing factor for transition, k>0
    """

    if (k <= 0.0):
        msg = ('k must be > 0.')
        raise ValueError(msg)

    f = _logit(np.log10(x), np.log10(x0), 1/k)
    n = (1.0-f)*alpha + f*beta
    return A*(x/x0)**n


def _logit(x, x0, k):
    """
    """
    return 1.0/(1.0+np.exp(-k*(x-x0)))


