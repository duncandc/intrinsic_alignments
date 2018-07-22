"""
cosmology utility functions
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np
from astropy import units as u
from astropy import constants as const
from hmf import growth_factor
import camb

__all__=('mean_density', 'linear_growth_factor', 'linear_power_spectrum', 'astropy_to_camb_cosmo')
__author__=('Duncan Campbell')

# define a default cosology for utilities
from astropy.cosmology import FlatLambdaCDM
default_cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.05, Tcmb0=2.7255)


def mean_density(z, cosmo=None):
    """
    Return the mean density of the Universe.

    Paramaters
    ----------
    z : array_like
        redshift

    cosmo : astropy.cosmology object

    Returns
    -------
    rho_b : numpy.array
         mean density of the universe at redshift z in units Msol/Mpc^3
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
    Return the growth factor, :math:`D(z)`, normalized to 1.0 at :math:`z=0`.

    Parmaters
    =========
    z : array_like

    cosmo : astropy.cosmology object
        if not specified, a defult cosmology is used.

    Returns
    =======
    d : numpy.array
        array of growth factors for each value of ``z``.
    """

    if cosmo is None:
        cosmo = default_cosmo

    # get growth factor function
    gf = growth_factor.GrowthFactor(cosmo)
    gf_f = gf.growth_factor
    return gf_f(z)


def linear_power_spectrum(z, cosmo=None, lmax=5000, minkh=1e-5, maxkh=100, npoints=1000):
    """
    Return a tabulated linear power spectrum, :math:`P(k)`.

    Parameters
    ==========
    z : float
        redshift for power spectrum

    cosmo : astropy.cosmology object
        if 'None', the default cosmology defined in cosmo_utils.py is used.

    Returns
    =======
    k : numpy.array
        tabulated values of k

    pk : numpy.array
        power spectrum at the specified k
    """

    # check to see if cosmology was passed
    if cosmo is None:
        cosmo = default_cosmo

    # The redshift argument for CAMB power spectrum must be a vector
    z = np.atleast_1d(z)
    # but this function is written for a single value of redshift
    if len(z)>1:
        msg = ('`z` parameter must be a float')
        raise ValueError(msg)

    # set up CAMB
    pars = camb.CAMBparams()
    cosmo_param_dict = astropy_to_camb_cosmo(cosmo)
    pars.set_cosmology(**cosmo_param_dict)
    pars.InitPower.set_params(ns=0.965, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    # calculate results for these parameters
    results = camb.get_results(pars)

    # note that non-linear corrections couple to small scales
    pars.set_matter_power(redshifts=z, kmax=maxkh*2)

    # retreive the linear spectrum
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=minkh,
                                                  maxkh=maxkh,
                                                  npoints=npoints)

    return kh, pk[0]


def astropy_to_camb_cosmo(cosmo):
    """
    Return a CAMB formatted cosmology dictionary given an Astropy cosmology object
    """

    # build a dictionary with CAMB keywords
    d = {
         'H0': cosmo.H0.value,
         'ombh2': cosmo.Ob0 * cosmo.h ** 2,
         'omch2': (cosmo.Om0 - cosmo.Ob0) * cosmo.h ** 2,
         'omk': cosmo.Ok0,
         'nnu': cosmo.Neff,
         'standard_neutrino_neff': cosmo.Neff,
         'TCMB': cosmo.Tcmb0.value  # value in Kelvin
         }

    return d





