r"""
models for galaxy desnity profiles
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.special import hyp2f1  # hypergeometric function
from scipy.special import betainc
from scipy.stats import beta


class cusped_density(object):
    """
    """

    def __init__(self, gamma=1, n=4, b_to_a=1.0, c_to_a=1.0):
        """
        """

        self.gamma = gamma
        self.n = n
        self.a = 1.0
        self.b = self.a * b_to_a
        self.c = self.a * c_to_a
        self.rho0 = 1.0

    def enclosed_mass(self, m):
        """
        """

        a = self.a
        b = self.b
        c = self.c
        gamma = self.gamma
        f2 = hyp2f1(self.n-gamma, 3-gamma, 4-gamma, m/(m+1.0))

        return 4.0*np.pi*a*b*c*(m/(m+1.0))**(3-gamma)*(1/(3-gamma))*f2

    def sample(self, size):
        """
        """

        if self.n > 3:
            beta.ppf(ranf,self.n-3, 3-self.gamma)


