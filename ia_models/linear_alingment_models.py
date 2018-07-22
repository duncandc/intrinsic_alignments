"""
linear models for intrinsic alignments
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import camb
import numpy as np
from scipy.interpolate import interp1d
from pyfftlog.pyfftlog import pk2xi
import scipy.integrate as integrate
from intrinsic_alignments.ia_models.cosmo_utils import linear_growth_factor, mean_density, linear_power_spectrum

from intrinsic_alignments.ia_models.cosmo_utils import default_cosmo


__author__=['Duncan Campbell']
__all__=['factor_PII', 'factor_PGI', 'xi_gg', 'ii_plus', 'ii_plus_projected']


def factor_PII(z, cosmo=None):
    """
    Return the II linear power spectrum multiplicative factor.
    See Bridle & King (2007), eq. 6.

    Paramaters
    ==========
    z : array_like
        redshift

    cosmo : astropy.cosmology object
        if 'None', the default cosmology defined in cosmo_utils.py is used.

    Returns
    =======
    A : float
        power spectrum factor
    """

    if cosmo is None:
        cosmo = default_cosmo

    C_1 = 5*10**(-14)
    return (C_1*mean_density(z, cosmo)/((1.0+z)*linear_growth_factor(z, cosmo)))**2


def factor_PGI(z, cosmo=None):
    """
    Return the GI linear power spectrum multiplicative factor.
    See Bridle & King (2007), eq. 11.

    Paramaters
    ==========
    z : array_like
        redshift

    cosmo : astropy.cosmology object
        if 'None', the default cosmology defined in cosmo_utils.py is used.

    Returns
    =======
    A : float
        power spectrum factor
    """

    if cosmo is None:
        cosmo = default_cosmo

    C_1 = 5*10**(-14)
    return -1.0*(C_1*mean_density(z, cosmo)/((1.0+z)*linear_growth_factor(z, cosmo)))


def xi_gg(r, z, cosmo=None):
    """
    Return the linear 3-D galaxy-galaxy clustering corrleation function, :math:`xi(r)`.

    Paramaters
    ==========
    r : array_like
        array of radial distances

    z : array_like
        redshift

    cosmo : astropy.cosmology object
        if 'None', the default cosmology defined in cosmo_utils.py is used.

    Returns
    =======
    xi : numpy.array
        linear galaxy-galaxy correlation function.
    """

    if cosmo is None:
        cosmo = default_cosmo

    kh, pk = linear_power_spectrum(z, cosmo=cosmo)
    tabulated_r, tabulated_xi = pk2xi(kh, pk)

    # interpolate between tabulated r and xi
    f_xi = interp1d(tabulated_r, tabulated_xi, fill_value='extrapolate')

    return f_xi(r), (kh, pk)


def ii_plus(r, z, cosmo=None):
    """
    Return the intrinsic–intrinsic (II) ellitpicity correlation function,
    :math:`\xi_{++}`.

    Paramaters
    ==========
    r : array_like
        array of radial distances

    z : array_like
        redshift

    cosmo : astropy.cosmology object
        if 'None', the default cosmology defined in cosmo_utils.py is used.

    Returns
    =======
    """

    if cosmo is None:
        cosmo = default_cosmo

    kh, pk = linear_power_spectrum(z, cosmo=cosmo)
    tabulated_r, tabulated_xi = pk2xi(kh, pk)

    # interpolate between r and xi
    f_xi = interp1d(tabulated_r, tabulated_xi*factor_PII(z, cosmo=cosmo), fill_value='extrapolate')

    return f_xi(r)


def ii_plus_projected(rp, z, pi_max=60.0, cosmo=None):
    """
    Return the projected intrinsic–intrinsic (II) ellitpicity correlation function,
    :math:`w_{++}`.

    Paramaters
    ==========
    r : array_like
        array of projected radial distances

    z : array_like
        redshift

    pi_max : float

    cosmo : astropy.cosmology object
        if 'None', the default cosmology defined in cosmo_utils.py is used.

    Returns
    =======
    """

    if cosmo is None:
        cosmo = default_cosmo

    # get interpolated integrand
    r = np.logspace(np.log10(np.min(rp)), np.log10(np.sqrt(pi_max**2 + np.max(rp)**2)), 100)
    xi = ii_plus(r, z, cosmo=cosmo)
    # interpolate between r and xi
    f_xi = interp1d(r, xi, fill_value='extrapolate')

    def integrand(x, y, z):
        r = np.sqrt(x*x + y*y)
        return 2.0*(f_xi(r))

    # integrate for each value of rp between pi=[0,pi_max].
    N = len(rp)
    result = np.zeros(N)
    for i in range(0, N):
        result[i] = integrate.quad(integrand, 0.0, pi_max, args=(rp[i],z,))[0]

    return result
