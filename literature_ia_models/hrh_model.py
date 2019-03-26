"""
classes for HRH models
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
import scipy.integrate as integrate


__author__=('Duncan Campbell')
__all__=('HRHstar', 'HRH')


class HRHstar(object):
    """
    modified Heavens, Refregier, and Heymans (HRH*) model class
    for modelling intrinsic galaxy alignment correlation functions
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ==========
        A : float
            amplitude for alignment correlations

        B : float
            saling length for alignbment correlations

        r0 : float
            correlation length for galaxy clustering model

        gamma : float
            galaxy clustering power law exponent

        R : float
            shear responsivity
        """

        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        Set the parameters for model.
        The values are taken from table 4 in Mandelbaum et al. 2006.
        """

        if 'sample' not in kwargs.keys():
            sample = 'SuperCOSMOS'
        else:
            sample = kwargs['sample']

        if sample == 'SDSS L3':
            self.params = {'A': -1.0*10**(-3),
                           'B': 1.0,
                           'r0': 5.25,
                           'gamma': 1.8,
                           'R': 0.87
                           }
        elif sample == 'SDSS L4':
            self.params = {'A': 1.7*10**(-3),
                           'B': 1.0,
                           'r0': 5.25,
                           'gamma': 1.8,
                           'R': 0.87
                           }
        elif sample == 'SDSS L5':
            self.params = {'A': 5.8*10**(-3),
                           'B': 1.0,
                           'r0': 5.25,
                           'gamma': 1.8,
                           'R': 0.87
                           }
        elif sample == 'SDSS L6':
            self.params = {'A': -43*10**(-3),
                           'B': 1.0,
                           'r0': 5.25,
                           'gamma': 1.8,
                           'R': 0.87
                           }
        elif sample == 'SDSS L3-L6':
            self.params = {'A': 1.8*10**(-3),
                           'B': 1.0,
                           'r0': 5.25,
                           'gamma': 1.8,
                           'R': 0.87
                           }
        elif sample == 'COMBO-17':
            self.params = {'A': 5.4*10**(-3),
                           'B': 1.0,
                           'r0': 5.25,
                           'gamma': 1.8,
                           'R': 0.87
                           }
        elif sample == 'SuperCOSMOS':
            self.params = {'A': 2.9*10**(-3),
                           'B': 1.0,
                           'r0': 5.25,
                           'gamma': 1.8,
                           'R': 0.87
                           }
        else:
            msg = ('sample for HRH model not recognized.')
            raise ValueError(msg)

        # if parameters are passed individually, over-ride tabulated values
        for key in kwargs.keys():
            if key in self.params.keys():
                self.params[key] = kwargs[key]

    def ee_3d(self, r):
        """
        A model for the ellipticity-ellipticity 3D
        ellipticity weighted correlation function, :math:`eta(r)`.

        Parameters
        ==========

        Returns
        =======
        """

        A = self.params['A']
        B = self.params['B']
        return A/(1.0+(r/B)**2.0)

    def gg_3d(self, r):
        """
        a model for the galaxy-galaxy 3D correlation function
        """

        return (r/self.params['r0'])**(-1.0*self.params['gamma'])

    def ii_plus_projected(self, rp, pi_max=60.0):
        """
        Return the projected intrinsic-intrinsic ellitpicity
        correlation function, math:`w_{++}`.
        """

        rp = np.atleast_1d(rp)

        def integrand(y, x):
            r = np.sqrt(x*x + y*y)
            return (1.0+self.gg_3d(r))*self.ee_3d(r)

        N = len(rp)
        result = np.zeros(N)

        # integrate for each value of rp
        for i in range(0, N):
            result[i] = integrate.quad(integrand, 0, pi_max, args=(rp[i],))[0]

        return 2.0*(1.0/(8.0*self.params['R']**2))*result


class HRH(object):
    """
    Heavens, Refregier, and Heymans (HRH) model class
    for modelling intrinsic galaxy alignment correlation functions
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ==========
        A : float
            amplitude for alignment correlations

        B : float
            saling length for alignbment correlations

        r0 : float
            correlation length for galaxy clustering model

        gamma : float
            galaxy clustering power law exponent

        R : float
            shear responsivity
        """

        self.set_params(**kwargs)

    def set_params(self, **kwargs):
        """
        Set the parameters for model. 
        The values are taken from eq. 9 in Heymans + 2004.
        """

        self.params = {'A': 0.012,
                       'B': 1.5,
                       'r0': 5.25,
                       'gamma': 1.8,
                       'R': 0.87
                      }

        # if parameters are passed individually, over-ride tabulated values
        for key in kwargs.keys():
            if key in self.params.keys():
                self.params[key] = kwargs[key]

    def ee_3d(self, r):
        """
        A model for the ellipticity-ellipticity 3D
        ellipticity weighted correlation function, :math:`eta(r)`.

        Parameters
        ==========

        Returns
        =======
        """

        A = self.params['A']
        B = self.params['B']
        return A*np.exp(-1.0*r/B)

    def gg_3d(self, r):
        """
        a model for the galaxy-galaxy 3D correlation function
        """

        return (r/self.params['r0'])**(-1.0*self.params['gamma'])

    def ii_plus_projected(self, rp, pi_max=60.0):
        """
        Return the projected intrinsic-intrinsic ellitpicity
        correlation function, math:`w_{++}`.
        """

        rp = np.atleast_1d(rp)

        def integrand(y, x):
            r = np.sqrt(x*x + y*y)
            return (1.0+self.gg_3d(r))*self.ee_3d(r)

        N = len(rp)
        result = np.zeros(N)

        # integrate for each value of rp
        for i in range(0, N):
            result[i] = integrate.quad(integrand, 0, pi_max, args=(rp[i],))[0]

        return 2.0*(1.0/(8.0*self.params['R']**2))*result
