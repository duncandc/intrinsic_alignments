r"""
models for galaxy desnity profiles
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.special import hyp2f1  # hypergeometric function
from scipy.special import betainc, beta
from scipy.special import btdtri as beta_inv


class cusped_density(object):
    """
    """

    def __init__(self, gamma=1, n=4, b_to_a=1.0, c_to_a=1.0, mtot=1.0):
        """
        """

        self.gamma = gamma
        self.n = n
        self.a = 1.0
        self.b = self.a * b_to_a
        self.c = self.a * c_to_a
        self.mtot = mtot
        self.rho0 = self.central_density()

    def enclosed_mass(self, m):
        """
        """

        a = self.a
        b = self.b
        c = self.c

        gamma = self.gamma
        n = self.n
        mtot = self.mtot

        rho0 = self.central_density()

        return 4.0*np.pi*a*b*c*rho0*betainc(n-3, 3-gamma, m/(m+1.0))*beta(n-3, 3-gamma)

    def central_density(self):
        """
        """
        
        a = self.a
        b = self.b
        c = self.c

        gamma = self.gamma
        n = self.n
        mtot = self.mtot

        return mtot/(4.0*np.pi*a*b*c*beta(n-3, 3-gamma))


    def sample(self, size):
        """
        returned axis algined particles drawn from density profile.
        - major axis is aligned with the x-axis
        - intermediate with the y-axis
        - minor axis with the z-axis
        """

        gamma = self.gamma
        n = self.n
        mtot = self.mtot

        a = self.a
        b = self.b
        c = self.c
        
        rho0 = self.central_density()
        k = 4.0*np.pi*a*b*c*beta(n-3, 3-gamma)     

        f = np.random.random(size)

        ff = beta_inv(n-3, 3-gamma, f*mtot/(rho0*k)) 
        m = -1.0*(ff/(ff-1.0))

        phi = np.random.uniform(0, 2*np.pi, size)
        cos_t = np.random.rand(size)*2 - 1
        sin_t = np.sqrt((1.0-cos_t*cos_t))

        x = a*m * np.cos(phi)
        y = b*m * np.sin(phi)
        z = c*m * cos_t

        return np.vstack((x,y,z)).T
        


