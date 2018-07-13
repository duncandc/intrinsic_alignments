"""
linear models for intrinsic alignments
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import camb
import numpy as np
from scipy.interpolate import interp1d
from pyfftlog.pyfftlog import pk2xi
import scipy.integrate as integrate
from intrinsic_alignments.ia_models.cosmo_utils import default_cosmo, linear_growth_factor, mean_density


def linear_power_spectrum(z, cosmo=None, lmax=2500, minkh=1e-4, maxkh=1, npoints=200):
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

    # redshift input for CAMB power spectrum must be a vector
    z = np.atleast_1d(z)
    # but this function is written for single values of the redshift
    if len(z)>1:
        msg = ('`z` parameter must be a float')
        raise ValueError(msg)

    # set up CAMB
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=cosmo.H0.value,
                       ombh2=cosmo.Ob0 * cosmo.h ** 2,
                       omch2=(cosmo.Om0 - cosmo.Ob0) * cosmo.h ** 2,
                       omk=cosmo.Ok0,
                       nnu=cosmo.Neff,
                       standard_neutrino_neff=cosmo.Neff,
                       TCMB=cosmo.Tcmb0.value)
    pars.InitPower.set_params(ns=0.965, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    # calculate results for these parameters
    results = camb.get_results(pars)

    # note the non-linear corrections couple to small scales
    pars.set_matter_power(redshifts=z, kmax=maxkh*2)

    # retreive linear spectrum
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=minkh,
                                                  maxkh=maxkh,
                                                  npoints=npoints)

    return kh, pk[0]


def P_II_factor(z, cosmo=None):
    """
    Return the II linear power spectrum multiplicative factor.
    See Bridle & King (2007) eq. 6.

    Paramaters
    ==========
    z : array_like
        redshift

    cosmo : astropy.cosmology object

    Returns
    =======
    A : float
        power spectrum factor
    """

    if cosmo is None:
        cosmo = default_cosmo

    C_1 = 5*10**(-14)
    return (C_1*mean_density(z, cosmo)/((1.0+z)*linear_growth_factor(z, cosmo)))**2


def ii_plus(r, z, cosmo=None, ptype='linear'):
    """
    """

    if cosmo is None:
        cosmo = default_cosmo

    kh, pk = linear_power_spectrum(z, cosmo=cosmo)
    tabulated_r, tabulated_xi = pk2xi(kh, pk)

    # interpolate between r and xi
    f_xi = interp1d(tabulated_r, tabulated_xi*P_II_factor(z, cosmo=cosmo), fill_value='extrapolate')

    return f_xi(r)


def ii_plus_projected(rp, z, pi_max=60.0, cosmo=None, ptype='linear'):
    """
    """

    if cosmo is None:
        cosmo = default_cosmo

    # get interpolated integrand
    r = np.logspace(np.min(rp), np.sqrt(pi_max**2 + np.max(rp)**2), 100)
    xi = ii_plus(r, z, cosmo=cosmo, ptype=ptype)
    # interpolate between r and xi
    f_xi = interp1d(r, xi, fill_value='extrapolate')

    def integrand(x, y, z):
        r = np.sqrt(x*x + y*y)
        return 2*(f_xi(r))

    # integrate for each value of rp
    N = len(rp)
    result = np.zeros(N)
    for i in range(0, N):
        result[i] = integrate.quad(integrand, 0.0, pi_max, args=(rp[i],z,))[0]

    return result
