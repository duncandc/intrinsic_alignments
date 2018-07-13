"""
cosmology utility functions
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy import units as u
from astropy import constants as const
from hmf import growth_factor

# define default cosology for utilities
from astropy.cosmology import FlatLambdaCDM
default_cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.7255)


def mean_density(z, cosmo=None):
    """
    mean density of the universe

    Paramaters
    ----------
    z : array_like
        redshift

    cosmo : astropy.cosmology object

    Returns
    -------
    rho_b : numpy.array
         mean density of the universe at redshift z in Msol/Mpc^3
    """

    if cosmo is None:
        cosmo = default_cosmo

    z = np.atleast_1d(z)
    a = 1.0/(1.0+z)  # scale factor

    rho = (3.0/(8.0*np.pi*const.G))*(cosmo.H(z)**2)*(cosmo.Om(z)*a**(-3.0))
    rho = rho.to(u.M_sun / u.parsec**3.0)*((10**6)**3.0)

    return rho.value


def linear_growth_factor(z, cosmo=None):
    """
    growth factor, :math:`D(z)`, normalized to 1.0 at :math:`z=0`.

    Parmaters
    =========
    z : array_like

    cosmo : astropy.cosmology object
        if not specified, a defult cosmology is used.

    Returns
    =======
    """

    if cosmo is None:
        cosmo = default_cosmo

    # get growth factor function
    gf = growth_factor.GrowthFactor(cosmo)
    gf_f = gf.growth_factor
    return gf_f(z)
