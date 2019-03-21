# -*- coding: utf-8 -*-
r"""
linear models for intrinsic alignments

Note that this module requires the pyfftlog package maintained on the McWilliams Center Github page
https://github.com/McWilliamsCenter/pyfftlog/blob/master/pyfftlog.py
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import camb
import numpy as np
from scipy.interpolate import interp1d
from pyfftlog.pyfftlog import pk2xi, call_transform
import scipy.integrate as integrate
from intrinsic_alignments.ia_models.cosmo_utils import linear_growth_factor, mean_density, linear_power_spectrum, nonlinear_power_spectrum

from intrinsic_alignments.ia_models.cosmo_utils import default_cosmo


__author__=['Duncan Campbell']
__all__=['LinearAlignmentModel', 'NonLinearAlignmentModel']


class LinearAlignmentModel(object):
    r"""
    Linear Intrinsic Alignment Model (LA)
    """

    def __init__(self, z=0.0, cosmo=None, **kwargs):
        r"""
        Paramaters
        ==========
        z : float
            redshift, default is 0.0

        cosmo : astropy.cosmology object
            default is set in cosmo_utils.py
        """

        if cosmo is None:
            cosmo = default_cosmo

        self.cosmo=cosmo
        self.z = z

        # get power spectrum
        kh, pk = linear_power_spectrum(z, cosmo=cosmo)
        self.power_spectrum = (kh, pk)
    
    def amplitude_ia(self):
        r"""
        Return the prefactor for IA used to scale the pwer specturm
        See Bridle & King (2007), eq. 6.

        Returns
        =======
        A : float
            power spectrum factor
        """

        C_1 = 5*10**(-14)
        A = (C_1*mean_density(self.z, self.cosmo)/((1.0+self.z)*linear_growth_factor(self.z, self.cosmo)))
        return A

    def xi_gg(self, r, b=1):
        r"""
        Return the linear 3-D galaxy-galaxy clustering auto-corrleation function, :math:`xi(r)`.

        Paramaters
        ==========
        r : array_like
            array of radial distances

        n : galaxy bias

        Returns
        =======
        xi : numpy.array
            linear galaxy-galaxy correlation function.
        """

        kh, pk = self.power_spectrum
        tabulated_r, tabulated_xi = pk2xi_0(kh, pk)

        # interpolate between tabulated r and xi
        f_xi = interp1d(tabulated_r, tabulated_xi, fill_value='extrapolate')

        # return the value of the interpolated correlation function at r
        return (b**2)*f_xi(r)

    def gi_plus(self, r):
        r"""
        Return the galaxy–intrinsic (GI) ellitpicity correlation function, :math:`\xi_{g+}(r)`.

        Paramaters
        ==========
        r : array_like
            array of radial distances

        Returns
        =======
        """
        
        # return the pre-computed correlation function
        kh, pk = self.power_spectrum
        tabulated_r, tabulated_xi_0 = pk2xi_2(kh, pk)

        # interpolate between r and xi
        A = self.amplitude_ia()
        f_xi = interp1d(tabulated_r, (tabulated_xi_2)*A, fill_value='extrapolate')

        # return the value of the interpolated correlation function at r
        return f_xi(r)

    def ii_plus(self, r):
        r"""
        Return the intrinsic–intrinsic (II) ellitpicity correlation function, :math:`\xi_{++}(r)`.

        Paramaters
        ==========
        r : array_like
            array of radial distances

        Returns
        =======
        """

        # return the pre-computed correlation function
        kh, pk = self.power_spectrum
        tabulated_r, tabulated_xi_0 = pk2xi_0(kh, pk)
        tabulated_r, tabulated_xi_4 = pk2xi_4(kh, pk)

        # interpolate between r and xi
        A = self.amplitude_ia()
        f_xi = interp1d(tabulated_r, (tabulated_xi_0+tabulated_xi_4)*(A**2), fill_value='extrapolate')

        # return the value of the interpolated correlation function at r
        return f_xi(r)


    def ii_cross(self, r):
        r"""
        Return the intrinsic–intrinsic (II) ellitpicity correlation function, :math:`\xi_{xx}(r)`.

        Paramaters
        ==========
        r : array_like
            array of radial distances

        Returns
        =======
        """

        # return the pre-computed correlation function
        kh, pk = self.power_spectrum
        tabulated_r, tabulated_xi_0 = pk2xi_0(kh, pk)
        tabulated_r, tabulated_xi_4 = pk2xi_4(kh, pk)

        # interpolate between r and xi
        A = self.amplitude_ia()
        f_xi = interp1d(tabulated_r, (tabulated_xi_0-tabulated_xi_4)*(A**2), fill_value='extrapolate')

        # return the value of the interpolated correlation function at r
        return f_xi(r)

    def ii_plus_projected(self, rp, pi_max=100.0, cosmo=None):
        r"""
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

        # get interpolated integrand
        r = np.logspace(np.log10(np.min(rp)), np.log10(np.sqrt(pi_max**2 + np.max(rp)**2)), 100)
        xi = self.ii_plus(r)
        # interpolate between r and xi
        f_xi = interp1d(r, xi, fill_value='extrapolate')

        def integrand(x, y):
            r = np.sqrt(x*x + y*y)
            return (f_xi(r))

        # integrate for each value of rp between pi=[0,pi_max].
        N = len(rp)
        result = np.zeros(N)
        for i in range(0, N):
            result[i] = 2.0*integrate.quad(integrand, 0.0, pi_max, args=(rp[i],))[0]

        return result

    def ii_cross_projected(self, rp, pi_max=100.0, cosmo=None):
        r"""
        Return the projected intrinsic–intrinsic (II) ellitpicity correlation function,
        :math:`w_{xx}`.

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

        # get interpolated integrand
        r = np.logspace(np.log10(np.min(rp)), np.log10(np.sqrt(pi_max**2 + np.max(rp)**2)), 100)
        xi = self.ii_cross(r)
        # interpolate between r and xi
        f_xi = interp1d(r, xi, fill_value='extrapolate')

        def integrand(x, y):
            r = np.sqrt(x*x + y*y)
            return (f_xi(r))

        # integrate for each value of rp between pi=[0,pi_max].
        N = len(rp)
        result = np.zeros(N)
        for i in range(0, N):
            result[i] = 2.0*integrate.quad(integrand, 0.0, pi_max, args=(rp[i],))[0]

        return result


class NonLinearAlignmentModel(LinearAlignmentModel):
    r"""
    Non-Linear Intrinsic Alignment Model (NLA)
    """

    def __init__(self, z=0.0, cosmo=None, **kwargs):
        """
        Paramaters
        ==========
        z : float
            redshift, default is 0.0

        cosmo : astropy.cosmology object
            default is set in cosmo_utils.py
        """

        LinearAlignmentModel.__init__(self)

        # replace the linear power specturm with the non-linear power spectrum
        kh, pk = nonlinear_power_spectrum(z, cosmo=cosmo)
        self.power_spectrum = (kh, pk)


#####################################################
##### simple wrappers around pyfftlog functions #####
#####################################################

def pk2xi_0(k, pk):
    r""" 
    """
    (r, xi) = call_transform(0, 2, k, pk, tdir=-1)
    return (r, xi)


def pk2xi_2(k, pk):
    r""" 
    """
    (r, xi) = call_transform(2, 2, k, pk, tdir=-1)
    return (r, xi)


def pk2xi_4(k, pk):
    r""" 
    """
    (r, xi) = call_transform(4, 2, k, pk, tdir=-1)
    return (r, xi)


