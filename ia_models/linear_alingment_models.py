"""
linear models for intrinsic alignments
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import camb
import numpy as np

from .cosmo_utils import default_cosmo


def linear_power_spectrum(z, cosmo=None, lmax=2500, minkh=1e-4, maxkh=1, npoints=200):
    """
    Return tabulated linear power spectrum(s), :math:`P(k)`.

    Parameters
    ==========
    z : float

    cosmo : astropy.cosmology object
        default uses the default cosmology dfined in cosmo_utils

    Returns
    =======
    k : numpy.array
        k for which the power spectrum is calculated

    pk : numpy.array
        power spectrum at the specified k
    """

    if cosmo is None:
        cosmo = default_cosmo

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

    # note non-linear corrections couple to small scales
    pars.set_matter_power(redshifts=z, kmax=maxkh*2)

    # linear spectra
    pars.NonLinear = camb.model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=npoints)

    return kh, pk


def P_II_factor(cosmo, z):
    """
    alignment shape-shape linear power spectrum factor
    """
    C_1 = 5*10**(-14)
    return (C_1*mean_density(cosmo, z)/((1.0+z)*linear_growth_factor(cosmo, z)))**2



