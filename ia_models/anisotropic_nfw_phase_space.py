"""
"""
from __future__ import division, print_function, absolute_import

import numpy as np
from astropy.table import Table

from halotools.empirical_models.phase_space_models.analytic_models.satellites.nfw.nfw_profile import NFWProfile
from halotools.empirical_models.phase_space_models.analytic_models.satellites.nfw.kernels import unbiased_dimless_vrad_disp as unbiased_dimless_vrad_disp_kernel

from halotools.empirical_models.phase_space_models.analytic_models.satellites.nfw.nfw_phase_space import NFWPhaseSpace
from halotools.empirical_models.phase_space_models.analytic_models.monte_carlo_helpers import MonteCarloGalProf

from halotools.empirical_models import model_defaults


__author__ = ['Andrew Hearin', 'Duncan Campbell']
__all__ = ['AnisotropicNFWPhaseSpace', 'MonteCarloAnisotropicGalProf']


class MonteCarloAnisotropicGalProf(MonteCarloGalProf):
    r"""
    sub-class of MonteCarloGalProf
    """

    def __init__(self):
        r"""
        """
        MonteCarloGalProf.__init__(self)


    def mc_unit_sphere(self, Npts, **kwargs):
        r""" 
        Returns Npts random points on the unit sphere.

        Parameters
        ----------
        Npts : int
            Number of 3d points to generate

        seed : int, optional
            Random number seed used in the Monte Carlo realization.
            Default is None, which will produce stochastic results.

        Returns
        -------
        x, y, z : array_like
            Length-Npts arrays of the coordinate positions.
        """
        seed = kwargs.get('seed', None)

        table = kwargs['table']

        npts = len(table)
        with NumpyRNGContext(seed):
            phi = np.random.uniform(0, 2*np.pi, Npts)
            uran = np.random.rand(npts)

        d = DimrothWatson()
        k = (1.0/table['halo_b_to_a']-1.0)
        cos_t = d.isf(uran, k)

        cos_t = d.isf(uran, k)
        sin_t = np.sqrt((1.-cos_t*cos_t))

        x = sin_t * np.cos(phi)
        y = sin_t * np.sin(phi)
        z = cos_t

        z_correlated_axes = np.vstack((x, y, z)).T

        z_axes = np.tile((0, 0, 1), npts).reshape((npts, 3))
        input_unit_vectors = np.vstack((table['halo_axisA_x'],
                                        table['halo_axisA_y'],
                                        table['halo_axisA_z'])).T

        angles = angles_between_list_of_vectors(z_axes, input_unit_vectors)
        rotation_axes = vectors_normal_to_planes(z_axes, input_unit_vectors)
        matrices = rotation_matrices_from_angles(angles, rotation_axes)

        correlated_axes = rotate_vector_collection(matrices, z_correlated_axes)
        x, y, z = correlated_axes[:, 0], correlated_axes[:, 1], correlated_axes[:, 2]
        return x, y, z


class AnisotropicNFWPhaseSpace(NFWPhaseSpace, MonteCarloAnisotropicGalProf):
    r"""
    sub-class of NFWPhaseSpace
    """
    def __init__(self, **kwargs):
        r"""
        Parameters
        ----------
        conc_mass_model : string or callable, optional
            Specifies the function used to model the relation between
            NFW concentration and halo mass.
            Can either be a custom-built callable function,
            or one of the following strings:
            ``dutton_maccio14``, ``direct_from_halo_catalog``.

        cosmology : object, optional
            Instance of an astropy `~astropy.cosmology`.
            Default cosmology is set in
            `~halotools.sim_manager.sim_defaults`.

        redshift : float, optional
            Default is set in `~halotools.sim_manager.sim_defaults`.

        mdef: str, optional
            String specifying the halo mass definition, e.g., 'vir' or '200m'.
            Default is set in `~halotools.empirical_models.model_defaults`.

        concentration_key : string, optional
            Column name of the halo catalog storing NFW concentration.

            This argument is only relevant when ``conc_mass_model``
            is set to ``direct_from_halo_catalog``. In such a case,
            the default value is ``halo_nfw_conc``,
            which is consistent with all halo catalogs provided by Halotools
            but may differ from the convention adopted in custom halo catalogs.

        concentration_bins : ndarray, optional
            Array storing how halo concentrations will be digitized when building
            a lookup table for mock-population purposes.
            The spacing of this array sets a limit on how accurately the
            concentration parameter can be recovered in a likelihood analysis.

        Examples
        --------
        >>> model = AnisotropicNFWPhaseSpace()
        """

        NFWPhaseSpace.__init__(self, **kwargs)
        AnisotropicMonteCarloGalProf.__init__(self)
        self.list_of_haloprops_needed = ['halo_b_to_a']


