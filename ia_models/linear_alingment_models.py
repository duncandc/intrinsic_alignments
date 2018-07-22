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
__all__=['LinearAlignmentModel', 'NonLinearAlignmentModel']


class LinearAlignmentModel(object):
    """
    Linear Intrinsic Alignment Model (LA)
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

        if cosmo is None:
            cosmo = default_cosmo

        self.cosmo=cosmo

        # get power spectrum
        kh, pk = linear_power_spectrum(z, cosmo=cosmo)
        tabulated_r, tabulated_xi = pk2xi(kh, pk)
        self.power_spectrum = (tabulated_r, tabulated_xi)


    def factor_PII(self):
        """
        Return the II linear power spectrum multiplicative factor.
        See Bridle & King (2007), eq. 6.

        Returns
        =======
        A : float
        power spectrum factor
        """

        C_1 = 5*10**(-14)
        return (C_1*mean_density(self.z, self.cosmo)/((1.0+self.z)*linear_growth_factor(self.z, self.cosmo)))**2


    def factor_PGI(self):
        """
        Return the GI linear power spectrum multiplicative factor.
        See Bridle & King (2007), eq. 11.

        Returns
        =======
        A : float
        power spectrum factor
        """

        C_1 = 5*10**(-14)
        return -1.0*(C_1*mean_density(self.z, self.cosmo)/((1.0+self.z)*linear_growth_factor(self.z, self.cosmo)))


    def xi_gg(self, r):
        """
        Return the linear 3-D galaxy-galaxy clustering corrleation function, :math:`xi(r)`.

        Paramaters
        ==========
        r : array_like
            array of radial distances

        Returns
        =======
        xi : numpy.array
        linear galaxy-galaxy correlation function.
        """

        kh, pk = self.power_spectrum
        tabulated_r, tabulated_xi = pk2xi(kh, pk)

        # interpolate between tabulated r and xi
        f_xi = interp1d(tabulated_r, tabulated_xi, fill_value='extrapolate')

        return f_xi(r), (kh, pk)


    def ii_plus(self, r):
        """
        Return the intrinsic–intrinsic (II) ellitpicity correlation function,
        :math:`\xi_{++}`.

        Paramaters
        ==========
        r : array_like
            array of radial distances

        Returns
        =======
        """

        kh, pk = self.power_spectrum
        tabulated_r, tabulated_xi = pk2xi(kh, pk)

        # interpolate between r and xi
        f_xi = interp1d(tabulated_r, tabulated_xi*self.factor_PII(), fill_value='extrapolate')

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

        # get interpolated integrand
        r = np.logspace(np.log10(np.min(rp)), np.log10(np.sqrt(pi_max**2 + np.max(rp)**2)), 100)
        xi = self.ii_plus(r)
        # interpolate between r and xi
        f_xi = interp1d(r, xi, fill_value='extrapolate')

        def integrand(x, y):
            r = np.sqrt(x*x + y*y)
            return 2.0*(f_xi(r))

        # integrate for each value of rp between pi=[0,pi_max].
        N = len(rp)
        result = np.zeros(N)
        for i in range(0, N):
            result[i] = integrate.quad(integrand, 0.0, pi_max, args=(rp[i],))[0]

        return result


class NonLinearAlignmentModel(LinearAlignmentModel):
    """
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

        # get power spectrum
        #kh, pk = nonlinear_power_spectrum(z, cosmo=cosmo)
        kh, pk = linear_power_spectrum(z, cosmo=cosmo)
        tabulated_r, tabulated_xi = pk2xi(kh, pk)
        self.power_spectrum = (tabulated_r, tabulated_xi)



