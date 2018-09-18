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
    A cusped desnity distribution with inner slope :math:`\gamma` 
    and outer slope :math:`n`  
    """

    def __init__(self, gamma=1, n=4, b_to_a=1.0, c_to_a=1.0, mtot=1.0):
        """
        Parameters
        ----------
        gamma : float
            inner slope

        n : float
            outer slop

        b_to_a : float
            intermediate to major axis ratio

        c_to_a : float
            minor to major axis ratio

        mtot : float
            total mass

        Notes
        -----
        gamma = 1 and n = 4, corresponds to the Hernquist profile.
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
        Return the enclosed mass within the dimensionless ellipsoidal radius, :math:`m`.

        Parameters
        ----------
        m : array_like
             array of dimensionless ellipsoidal radius
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
        Return central desnity, :math:`\rho_0`.
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
        Returned Monte Carlo samples from the 3D triaxial density distribution.
        
        Parameters
        ----------
        size: int
            number of samples to return

        Notes
        -----
        The distirbution is assumed to be axis-aligned with the
        major axis aligned with the x-axis, the intermediate with 
        the y-axisv and the minor axis with the z-axis.
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
        


