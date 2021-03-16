r"""
halotools model components for modelling central and scatellite intrinsic alignments
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np

# vector rotations
from rotations import rotate_vector_collection
from rotations.mcrotations import random_perpendicular_directions, random_unit_vectors_3d
from rotations.vector_utilities import (elementwise_dot, elementwise_norm, normalized_vectors,
                                        angles_between_list_of_vectors)
from rotations.rotations3d import (vectors_between_list_of_vectors, vectors_normal_to_planes,
                                   rotation_matrices_from_angles)
# watson distribution
from watson_dist import DimrothWatson

# utilities
from warnings import warn
from astropy.utils.misc import NumpyRNGContext


__all__ = ('RandomAlignment',
           'CentralAlignment',
           'SatelliteAlignment',
           'RadialSatelliteAlignment',
           'MajorAxisSatelliteAlignment',
           'HybridSatelliteAlignment',
           'HaloMassCentralAlignmentStrength',
           'RadialSatelliteAlignmentStrength')
__author__ = ('Duncan Campbell',)


class RandomAlignment(object):
    """
    class to model random galaxy orientations
    """
    def __init__(self, gal_type='centrals', **kwargs):
        """
        """

        self.gal_type = gal_type

        self._mock_generation_calling_sequence = (['assign_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = []
        self._methods_to_inherit = ([])
        self.param_dict = ({})

    def assign_orientation(self, **kwargs):
        r"""
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            N = len(table)
        else:
            N = kwargs['size']

        # assign random orientations
        major_v = random_unit_vectors_3d(N)
        inter_v = random_perpendicular_directions(major_v)
        minor_v = normalized_vectors(np.cross(major_v, inter_v))

        if 'table' in kwargs.keys():

            try:
                mask = (table['gal_type'] == self.gal_type)
            except KeyError:
                mask = np.array([True]*N)
                msg = ("Because `gal_type` is not indicated in `table`, the orientations",
                       "are being assigned for all galaxies in the `table`.")
                print(msg)

            # check to see if the columns exist
            for key in list(self._galprop_dtypes_to_allocate.names):
                if key not in table.keys():
                    table[key] = 0.0

            table['galaxy_axisA_x'][mask] = major_v[mask, 0]
            table['galaxy_axisA_y'][mask] = major_v[mask, 1]
            table['galaxy_axisA_z'][mask] = major_v[mask, 2]

            table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
            table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
            table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

            table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
            table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
            table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

            return table
        else:
            return major_v, inter_v, minor_v


class CentralAlignment(object):
    r"""
    alignment model for central galaxies in host-haloes
    """
    def __init__(self, central_alignment_strength=1.0, prim_gal_axis='major', **kwargs):
        r"""
        Parameters
        ----------
        central_alignment_strength : float
            [-1,1] bounded number indicating alignment strength

        prim_gal_axis :  string, optional
            string indicating which galaxy principle axis is correlated with the halo alignment axis.
            The options are: `major`, `intermediate`, and `minor`.

        alignment_keys : list
            A list of strings indicating the keywords for the x,y- and z components of the
            halo alignment vector. Deafult is ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z'].

        Notes
        -----
        If the kwargs or table contains a key "alignment_strength", when populating a mock,
        this will be used instead of the `central_alignment_stregth` parameter passed during intialization.
        This is how varying the alignment strength as a function of galaxy/halo properties is handeled.
        """

        self.gal_type = 'centrals'
        self._mock_generation_calling_sequence = (['assign_central_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        # specify the halo alignment vector
        if 'alignment_keys' in kwargs.keys():
            assert len(kwargs['alignment_keys'])==3
            self.list_of_haloprops_needed = kwargs['alignment_keys']
        else:
            self.list_of_haloprops_needed = ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']

        # set which galaxy axis is correlated with the halo alignment vector
        possible_axis = ['major', 'intermediate', 'minor']
        if prim_gal_axis in possible_axis:
            if prim_gal_axis == possible_axis[0]: self.prim_gal_axis = 'A'
            elif prim_gal_axis == possible_axis[1]: self.prim_gal_axis = 'B'
            elif prim_gal_axis == possible_axis[2]: self.prim_gal_axis = 'C'
        else:
            msg = ('`prim_gal_axis` must be one of {0}, but instead is {1}.'.format(possible_axis, prim_gal_axis))
            raise ValueError(msg)

        self._methods_to_inherit = (
            ['assign_central_orientation'])
        self.param_dict = ({
            'central_alignment_strength': central_alignment_strength})

    def assign_central_orientation(self, **kwargs):
        r"""
        Assign a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis

        Parameters
        ==========
        halo_axisA_x, halo_axisA_y, halo_axisA_z :  array_like
             x,y,z components of halo alignment axis

        Returns
        =======
        major_aixs, intermediate_axis, minor_axis :  numpy nd.arrays
            arrays of galaxies' axes
        """
        if 'table' in kwargs.keys():
            table = kwargs['table']
            Ax = table[self.list_of_haloprops_needed[0]]
            Ay = table[self.list_of_haloprops_needed[1]]
            Az = table[self.list_of_haloprops_needed[2]]
        else:
            Ax = kwargs['halo_axisA_x']
            Ay = kwargs['halo_axisA_y']
            Az = kwargs['halo_axisA_z']

        # get alignment strength for each galaxy
        if 'table' in kwargs.keys():
            try:
                p = table['central_alignment_strength']
            except KeyError:
                msg = ('`central_alignment_strength` not detected in the table, using value in self.param_dict.')
                warn(msg)
                p = np.ones(len(Ax))*self.param_dict['central_alignment_strength']
        else:
            p = np.ones(len(Ax))*self.param_dict['central_alignment_strength']

        # set prim_gal_axis orientation
        major_input_vectors = np.vstack((Ax, Ay, Az)).T
        A_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

        # randomly set secondary axis orientation
        B_v = random_perpendicular_directions(A_v)

        # the tertiary axis is determined
        C_v = vectors_normal_to_planes(A_v, B_v)

        # depending on the prim_gal_axis, assign correlated axes
        if self.prim_gal_axis == 'A':
            major_v = A_v
            inter_v = B_v
            minor_v = C_v
        elif self.prim_gal_axis == 'B':
            major_v = B_v
            inter_v = A_v
            minor_v = C_v
        elif self.prim_gal_axis == 'C':
            major_v = B_v
            inter_v = C_v
            minor_v = A_v
        else:
            msg = ('primary galaxy axis {0} is not recognized.'.format(self.prim_gal_axis))
            raise ValueError(msg)

        if 'table' in kwargs.keys():
            try:
                mask = (table['gal_type'] == self.gal_type)
            except KeyError:
                mask = np.array([True]*len(table))
                msg = ("Because `gal_type` not indicated in `table`.",
                       "The orientation is being assigned for all galaxies in the `table`.")
                print(msg)

            # check to see if the columns exist
            for key in list(self._galprop_dtypes_to_allocate.names):
                if key not in table.keys():
                    table[key] = 0.0

            # add orientations to the galaxy table
            table['galaxy_axisA_x'][mask] = major_v[mask, 0]
            table['galaxy_axisA_y'][mask] = major_v[mask, 1]
            table['galaxy_axisA_z'][mask] = major_v[mask, 2]

            table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
            table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
            table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

            table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
            table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
            table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

            return table
        else:
            return major_v, inter_v, minor_v


class SatelliteAlignment(object):
    r"""
    alignment model for satellite galaxies in sub-haloes
    """
    def __init__(self, satellite_alignment_strength=1.0, prim_gal_axis='major', **kwargs):
        r"""
        Parameters
        ----------
        satellite_alignment_strength : float
            [-1,1] bounded number indicating alignment strength

        prim_gal_axis :  string, optional
            string indicating which galaxy principle axis is correlated with the halo alignment axis.
            The options are: `major`, `intermediate`, and `minor`.

        alignment_keys : list
            A list of strings indicating the keywords for the x,y- and z components of the
            halo alignment vector. Deafult is ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z'].

        Notes
        -----
        If the kwargs or table contains a key "alignment_strength", when populating a mock,
        this will be used instead of the `satellite_alignment_stregth` parameter passed during intialization.
        This is how varying the alignment strength as a function of galaxy/halo properties is handeled.
        """

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['assign_satellite_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        # specify the halo alignment vector
        if 'alignment_keys' in kwargs.keys():
            assert len(kwargs['alignment_keys'])==3
            self.list_of_haloprops_needed = kwargs['alignment_keys']
        else:
            self.list_of_haloprops_needed = ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']

        # set which galaxy axis is correlated with the halo alignment vector
        possible_axis = ['major', 'intermediate', 'minor']
        if prim_gal_axis in possible_axis:
            if prim_gal_axis == possible_axis[0]: self.prim_gal_axis = 'A'
            elif prim_gal_axis == possible_axis[1]: self.prim_gal_axis = 'B'
            elif prim_gal_axis == possible_axis[2]: self.prim_gal_axis = 'C'
        else:
            msg = ('`prim_gal_axis` must be one of {0}, but instead is {1}.'.format(possible_axis, prim_gal_axis))
            raise ValueError(msg)

        self._methods_to_inherit = (
            ['assign_satellite_orientation'])
        self.param_dict = ({
            'satellite_alignment_strength': satellite_alignment_strength})

    def assign_satellite_orientation(self, **kwargs):
        r"""
        Assign a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis

        Parameters
        ==========
        halo_axisA_x, halo_axisA_y, halo_axisA_z :  array_like
             x,y,z components of halo alignment axis

        Returns
        =======
        major_aixs, intermediate_axis, minor_axis :  numpy nd.arrays
            arrays of galaxies' axes
        """
        if 'table' in kwargs.keys():
            table = kwargs['table']
            Ax = table[self.list_of_haloprops_needed[0]]
            Ay = table[self.list_of_haloprops_needed[1]]
            Az = table[self.list_of_haloprops_needed[2]]
        else:
            Ax = kwargs['halo_axisA_x']
            Ay = kwargs['halo_axisA_y']
            Az = kwargs['halo_axisA_z']

        # get alignment strength for each galaxy
        if 'table' in kwargs.keys():
            try:
                p = table['satellite_alignment_strength']
            except KeyError:
                msg = ('`satellite_alignment_strength` not detected in the table, using value in self.param_dict.')
                warn(msg)
                p = np.ones(len(Ax))*self.param_dict['satellite_alignment_strength']
        else:
            p = np.ones(len(Ax))*self.param_dict['satellite_alignment_strength']

        # set prim_gal_axis orientation
        major_input_vectors = np.vstack((Ax, Ay, Az)).T
        A_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

        # randomly set secondary axis orientation
        B_v = random_perpendicular_directions(A_v)

        # the tertiary axis is determined
        C_v = vectors_normal_to_planes(A_v, B_v)

        # depending on the prim_gal_axis, assign correlated axes
        if self.prim_gal_axis == 'A':
            major_v = A_v
            inter_v = B_v
            minor_v = C_v
        elif self.prim_gal_axis == 'B':
            major_v = B_v
            inter_v = A_v
            minor_v = C_v
        elif self.prim_gal_axis == 'C':
            major_v = B_v
            inter_v = C_v
            minor_v = A_v
        else:
            msg = ('primary galaxy axis {0} is not recognized.'.format(self.prim_gal_axis))
            raise ValueError(msg)

        if 'table' in kwargs.keys():
            try:
                mask = (table['gal_type'] == self.gal_type)
            except KeyError:
                mask = np.array([True]*len(table))
                msg = ("Because `gal_type` not indicated in `table`.",
                       "The orientation is being assigned for all galaxies in the `table`.")
                print(msg)

            # check to see if the columns exist
            for key in list(self._galprop_dtypes_to_allocate.names):
                if key not in table.keys():
                    table[key] = 0.0

            # add orientations to the galaxy table
            table['galaxy_axisA_x'][mask] = major_v[mask, 0]
            table['galaxy_axisA_y'][mask] = major_v[mask, 1]
            table['galaxy_axisA_z'][mask] = major_v[mask, 2]

            table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
            table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
            table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

            table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
            table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
            table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

            return table
        else:
            return major_v, inter_v, minor_v


class HaloMassCentralAlignmentStrength():
    """
    model for the stregth of alignment for centrals
    """

    def __init__(self, central_alignment_a=0.05, central_alignment_gamma=0.25):
        """
        Parameters
        ==========
        a : float

        alpha : float
        """

        self.gal_type = 'centrals'
        self._mock_generation_calling_sequence = (['assign_central_alignment_strength'])

        self._galprop_dtypes_to_allocate = np.dtype([(str('central_alignment_strength'), 'f4')])

        self.list_of_haloprops_needed = ['halo_mvir']

        self._methods_to_inherit = (['assign_central_alignment_strength'])
        self.param_dict = ({
            'a': central_alignment_a,
            'gamma': central_alignment_gamma})

    def assign_central_alignment_strength(self, **kwargs):
        """
        Parameters
        ==========
        halo_mvir : array_like
            host halo virial mass
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            halo_m = table['halo_mvir']
        else:
            halo_m = kwargs['halo_mvir']

        s = self.alignment_strength_mass_dependence(halo_m)

        if 'table' in kwargs.keys():
            mask = (table['gal_type'] == self.gal_type)
            table['central_alignment_strength'] = 0.0
            table['central_alignment_strength'][mask] = s[mask]
            return table
        else:
            return s

    def alignment_strength_mass_dependence(self, m):
        """
        Parameters
        ==========
        m : array_like
            scaled halo masses

        Returns
        =======
        alignment_strength : numpy.array
            array fo values bounded between [-1,1]
        """

        a = self.param_dict['a']
        gamma = self.param_dict['gamma']
        result = a*np.log10(m)+gamma
        mask = (result < -0.99)
        result[mask]= -0.99
        mask = (result > 0.99)
        result[mask]= 0.99
        return result


class RadialSatelliteAlignment(object):
    r"""
    radial alignment model for satellite galaxies
    """

    def __init__(self, prim_gal_axis='major', **kwargs):
        """
        Parameters
        ----------
        satellite_alignment_strength : float
            parameter between [-1,1] that sets the alignment strength

        prim_gal_axis :  string, optional
            string indicating which galaxy principle axis is correlated with the halo alignment axis.
            The options are: `major`, `intermediate`, or `minor`.

        Notes
        -----
        If the kwargs or table contain a key "satellite_alignment_strength", this will be used instead.
        """

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['inherit_halocat_properties',
                                                   'assign_satellite_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = ['halo_x', 'halo_y', 'halo_z', 'halo_rvir']

        # set which galaxy axis is correlated with the halo alignment vector
        possible_axis = ['major', 'intermediate', 'minor']
        if prim_gal_axis in possible_axis:
            if prim_gal_axis == possible_axis[0]: self.prim_gal_axis = 'A'
            elif prim_gal_axis == possible_axis[1]: self.prim_gal_axis = 'B'
            elif prim_gal_axis == possible_axis[2]: self.prim_gal_axis = 'C'
        else:
            msg = ('`prim_gal_axis` must be one of {0}, but instead is {1}.'.format(possible_axis, prim_gal_axis))
            raise ValueError(msg)

        # set default box size.
        if 'Lbox' in kwargs.keys():
            self._Lbox = kwargs['Lbox']
        else:
            self._Lbox = np.inf
        # update Lbox if a halo catalog object is passed.
        self._additional_kwargs_dict = dict(inherit_halocat_properties=['Lbox'])

        self._methods_to_inherit = (
            ['assign_satellite_orientation', 'inherit_halocat_properties'])

        # set parameters
        self.set_default_params()
        if 'satellite_alignment_strength' in kwargs.keys():
            mu_sat = kwargs['satellite_alignment_strength']
            self.param_dict = ({'satellite_alignment_strength': mu_sat})

    def set_default_params(self):
        r"""
        set default parameters
        """
        d = {'satellite_alignment_strength': 0.8}
        self.param_dict = d

    def inherit_halocat_properties(self, seed=None, **kwargs):
        r"""
        inherit the box size during mock population
        """
        Lbox = kwargs['Lbox']
        self._Lbox = Lbox

    def assign_satellite_orientation(self, **kwargs):
        r"""
        assign a a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis

        Returns
        -------
        major_aixs, intermediate_axis, minor_axis :  numpy nd.arrays
            arrays of galaxies' axies
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            try:
                Lbox = kwargs['Lbox']
            except KeyError:
                Lbox = self._Lbox
        else:
            try:
                Lbox = kwargs['Lbox']
            except KeyError:
                Lbox = self._Lbox

        # calculate the radial vector between satellites and centrals
        major_input_vectors, r = self.get_radial_vector(Lbox=Lbox, **kwargs)

        # check for length 0 radial vectors
        mask = (r<=0.0) | (~np.isfinite(r))
        N_bad_axes = np.sum(mask)
        if N_bad_axes>0:
            major_input_vectors[mask,:] = random_unit_vectors_3d(N_bad_axes)
            msg = ('{0} galaxies have a radial distance equal to zero (or infinity) from their host. '
                   'These galaxies will be re-assigned random alignment vectors.'.format(int(N_bad_axes)))
            warn(msg)

        # get alignment strength for each galaxy
        if 'table' in kwargs.keys():
            try:
                p = table['satellite_alignment_strength']
            except KeyError:
                msg = ('`satellite_alignment_strength` key not detected in `table`.'
                       'The value set in self.param_dict of this class will be used instead.')
                warn(msg)
                p = np.ones(len(table))*self.param_dict['satellite_alignment_strength']
        else:
            N = len(self.param_dict['x'])
            p = np.ones(N*self.param_dict['satellite_alignment_strength'])

        # set prim_gal_axis orientation
        A_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

        # check for nan vectors
        mask = (~np.isfinite(np.sum(np.prod(A_v, axis=-1))))
        N_bad_axes = np.sum(mask)
        if N_bad_axes>0:
            A_v[mask,:] = random_unit_vectors_3d(N_bad_axes)
            msg = ('{0} correlated alignment axis(axes) were found to be not finite. '
                   'These will be re-assigned random vectors.'.format(int(N_bad_axes)))
            warn(msg)

        # randomly set secondary axis orientation
        B_v = random_perpendicular_directions(A_v)

        # the tertiary axis is determined
        C_v = vectors_normal_to_planes(A_v, B_v)

        # depending on the prim_gal_axis, assign correlated axes
        if self.prim_gal_axis == 'A':
            major_v = A_v
            inter_v = B_v
            minor_v = C_v
        elif self.prim_gal_axis == 'B':
            major_v = B_v
            inter_v = A_v
            minor_v = C_v
        elif self.prim_gal_axis == 'C':
            major_v = B_v
            inter_v = C_v
            minor_v = A_v

        if 'table' in kwargs.keys():
            try:
                mask = (table['gal_type'] == self.gal_type)
            except KeyError:
                mask = np.array([True]*len(table))
                msg = ("`gal_type` not indicated in `table`.",
                       "The orientation is being assigned for all galaxies in the `table`.")
                print(msg)

            # check to see if the columns exist
            for key in list(self._galprop_dtypes_to_allocate.names):
                if key not in table.keys():
                    table[key] = 0.0

            # add orientations to the galaxy table
            table['galaxy_axisA_x'][mask] = major_v[mask, 0]
            table['galaxy_axisA_y'][mask] = major_v[mask, 1]
            table['galaxy_axisA_z'][mask] = major_v[mask, 2]

            table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
            table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
            table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

            table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
            table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
            table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

            return table
        else:
            return major_v, inter_v, minor_v


    def get_radial_vector(self, Lbox=None, **kwargs):
        """
        caclulate the radial vector for satellite galaxies

        Parameters
        ==========
        x, y, z : array_like
            galaxy positions

        halo_x, halo_y, halo_z : array_like
            host halo positions

        halo_r : array_like
            halo size

        Lbox : array_like
            array len(3) giving the simulation box size along each dimension

        Returns
        =======
        r_vec : numpy.array
            array of radial vectors of shape (Ngal, 3) between host haloes and satellites

        r : numpy.array
            radial distance
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            x = table['x']
            y = table['y']
            z = table['z']
            halo_x = table['halo_x']
            halo_y = table['halo_y']
            halo_z = table['halo_z']
        else:
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            halo_x = kwargs['halo_x']
            halo_y = kwargs['halo_y']
            halo_z = kwargs['halo_z']

        if Lbox is None:
            Lbox = self._Lbox

        # define halo-center - satellite vector
        # accounting for PBCs
        dx = (x - halo_x)
        mask = dx>Lbox[0]/2.0
        dx[mask] = dx[mask] - Lbox[0]
        mask = dx<-1.0*Lbox[0]/2.0
        dx[mask] = dx[mask] + Lbox[0]

        dy = (y - halo_y)
        mask = dy>Lbox[1]/2.0
        dy[mask] = dy[mask] - Lbox[1]
        mask = dy<-1.0*Lbox[1]/2.0
        dy[mask] = dy[mask] + Lbox[1]

        dz = (z - halo_z)
        mask = dz>Lbox[2]/2.0
        dz[mask] = dz[mask] - Lbox[2]
        mask = dz<-1.0*Lbox[2]/2.0
        dz[mask] = dz[mask] + Lbox[2]

        r_vec = np.vstack((dx, dy, dz)).T
        r = np.sqrt(np.sum(r_vec*r_vec, axis=-1))

        return r_vec, r


class RadialSatelliteAlignmentStrength():
    """
    model for the stregth of alignment of satellites
    """

    def __init__(self,  satellite_alignment_a= 0.00650463, satellite_alignment_gamma=-0.04322356):
        """
        Parameters
        ==========
        a : float

        alpha : float
        """

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['assign_satellite_alignment_strength'])

        self._galprop_dtypes_to_allocate = np.dtype([(str('satellite_alignment_strength'), 'f4')])

        self.list_of_haloprops_needed = ['halo_x', 'halo_y', 'halo_z', 'halo_rvir']

        self._additional_kwargs_dict = dict(inherit_halocat_properties=['Lbox'])

        self._methods_to_inherit = (['assign_satellite_alignment_strength'])
        self.param_dict = ({
            'a': satellite_alignment_a,
            'gamma': satellite_alignment_gamma})

    def inherit_halocat_properties(self, **kwargs):
        """
        """
        Lbox = kwargs['Lbox']
        self._Lbox = Lbox

    def assign_satellite_alignment_strength(self, **kwargs):
        """
        Parameters
        ==========
        x, y, z : array_like
            galaxy positions

        halo_x, halo_y, halo_z : array_like
            host halo positions

        halo_r : array_like
            host halo virial radius

        Lbox : array_like
            size of simulation along each dimension
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            x = table['x']
            y = table['y']
            z = table['z']
            halo_x = table['halo_x']
            halo_y = table['halo_y']
            halo_z = table['halo_z']
            halo_r = table['halo_rvir']
            try:
                Lbox = kwargs['Lbox']
            except KeyError:
                Lbox = self._Lbox
        else:
            x = kwargs['x']
            y = kwargs['y']
            z = kwargs['z']
            halo_x = kwargs['halo_x']
            halo_y = kwargs['halo_y']
            halo_z = kwargs['halo_z']
            halo_r = kwargs['halo_rvir']
            Lbox = kwargs['Lbox']

        # define halo-center - satellite vector
        dx = (x - halo_x)
        mask = dx>Lbox[0]/2.0
        dx[mask] = dx[mask] - Lbox[0]
        mask = dx<-1.0*Lbox[0]/2.0
        dx[mask] = dx[mask] + Lbox[0]

        dy = (y - halo_y)
        mask = dy>Lbox[1]/2.0
        dy[mask] = dy[mask] - Lbox[1]
        mask = dy<-1.0*Lbox[1]/2.0
        dy[mask] = dy[mask] + Lbox[1]

        dz = (z - halo_z)
        mask = dz>Lbox[2]/2.0
        dz[mask] = dz[mask] - Lbox[2]
        mask = dz<-1.0*Lbox[2]/2.0
        dz[mask] = dz[mask] + Lbox[2]

        # calculate scaled halo virial radius
        r = np.sqrt(dx**2 + dy**2 + dz**2)/halo_r

        s = self.alignment_strength_radial_dependence(r)

        if 'table' in kwargs.keys():
            mask = (table['gal_type'] == self.gal_type)
            table['satellite_alignment_strength'] = 0.0
            table['satellite_alignment_strength'][mask] = s[mask]
            return table
        else:
            return s

    def _alignment_strength_radial_dependence(self, r):
        """
        Parameters
        ==========
        r : array_like
            scaled radial position

        Returns
        =======
        alignment_strength : numpy.array
            array fo values bounded between [-1,1]
        """

        r = np.atleast_1d(r)

        a = self.param_dict['a']
        gamma = self.param_dict['gamma']

        with np.errstate(divide='ignore', invalid='ignore'):
            result = np.where(r!=0, a*(1.0-1.0/(1.0+(1.0/r)**gamma)), 0.99)

        mask = (result < -0.99)
        result[mask]= -0.99
        mask = (result > 0.99)
        result[mask]= 0.99

        return result

    def alignment_strength_radial_dependence(self, r):
        """
        Parameters
        ==========
        r : array_like
            scaled radial position

        Returns
        =======
        alignment_strength : numpy.array
            array fo values bounded between [-1,1]
        """

        r = np.atleast_1d(r)

        a = self.param_dict['a']
        gamma= self.param_dict['gamma']

        ymax = 0.99
        ymin = -0.99

        result = np.zeros(len(r))
        result = (r/a)**gamma

        mask = (result > ymax)
        result[mask] = ymax

        mask = (result < ymin)
        result[mask] = ymin

        return result



class MajorAxisSatelliteAlignment(object):
    r"""
    alignment model for satellite galaxies
    """
    def __init__(self, satellite_alignment_strength=0.8, prim_gal_axis='major', **kwargs):
        """
        Parameters
        ==========
        satellite_alignment_strength : float
             parameter between [-1,1] that sets the alignment strength between perfect anti-alignment and perfect alignment

        prim_gal_axis :  string, optional
            string indicating which galaxy principle axis is correlated with the halo alignment axis.
            The options are: `major`, `intermediate`, and `minor`

        Notes
        =====
        If the kwargs or table contain a key "satellite_alignment_strength", this will be used instead.
        """

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['assign_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = ['halo_x', 'halo_y', 'halo_z', 'halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']

        # set which galaxy axis is correlated with the halo alignment vector
        possible_axis = ['major', 'intermediate', 'minor']
        if prim_gal_axis in possible_axis:
            if prim_gal_axis == possible_axis[0]: self.prim_gal_axis = 'A'
            elif prim_gal_axis == possible_axis[1]: self.prim_gal_axis = 'B'
            elif prim_gal_axis == possible_axis[2]: self.prim_gal_axis = 'C'
        else:
            msg = ('`prim_gal_axis` muyst be one of {0}, but instead is {1}.'.format(possible_axis, prim_gal_axis))
            raise ValueError(msg)

        self._methods_to_inherit = (
            ['assign_orientation'])
        self.param_dict = ({
            'satellite_alignment_a1': satellite_alignment_a1,
            'satellite_alignment_alpha1': satellite_alignment_alpha1})

    def assign_orientation(self, **kwargs):
        r"""
        assign a a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis
        """
        if 'table' in kwargs.keys():
            table = kwargs['table']
            halo_x = table['halo_x']
            halo_y = table['halo_y']
            halo_z = table['halo_z']
            Ax = table[self.list_of_haloprops_needed[3]]
            Ay = table[self.list_of_haloprops_needed[4]]
            Az = table[self.list_of_haloprops_needed[5]]
        else:
            halo_x = kwargs['halo_x']
            halo_y = kwargs['halo_y']
            halo_z = kwargs['halo_z']
            Ax = kwargs['halo_axisA_x']
            Ay = kwargs['halo_axisA_y']
            Az = kwargs['halo_axisA_z']

        # get alignment strength for each galaxy
        if 'table' in kwargs.keys():
            try:
                p = table['satellite_alignment_strength']
            except KeyError:
                msg = ('`satellite_alignment_strength` not detected in the table, using value in self.param_dict.')
                warn(msg)
                p = np.ones(len(table))*self.param_dict['satellite_alignment_strength']
        else:
            N = len(self.param_dict['x'])
            p = np.ones(N*self.param_dict['satellite_alignment_strength'])

        # set halo alignment vector
        major_input_vectors = np.vstack((Ax, Ay, Az)).T

        # set prim_gal_axis orientation
        A_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

        # randomly set secondary axis orientation
        B_v = random_perpendicular_directions(A_v)

        # the tertiary axis is determined
        C_v = vectors_normal_to_planes(A_v, B_v)

        # depending on the prim_gal_axis, assign correlated axes
        if self.prim_gal_axis == 'A':
            major_v = A_v
            inter_v = B_v
            minor_v = C_v
        elif self.prim_gal_axis == 'B':
            major_v = B_v
            inter_v = A_v
            minor_v = C_v
        elif self.prim_gal_axis == 'C':
            major_v = B_v
            inter_v = C_v
            minor_v = A_v

        if 'table' in kwargs.keys():
            try:
                mask = (table['gal_type'] == self.gal_type)
            except KeyError:
                mask = np.array([True]*len(table))
                msg = ("`gal_type` not indicated in `table`.",
                       "The orientation is being assigned for all galaxies in the `table`.")
                print(msg)

            # add orientations to the galaxy table
            table['galaxy_axisA_x'][mask] = major_v[mask, 0]
            table['galaxy_axisA_y'][mask] = major_v[mask, 1]
            table['galaxy_axisA_z'][mask] = major_v[mask, 2]

            table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
            table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
            table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

            table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
            table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
            table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

            return table
        else:
            return major_v, inter_v, minor_v


class HybridSatelliteAlignment(object):
    r"""
    alignment model for satellite galaxies
    """
    def __init__(self, satellite_alignment_strength=0.8, satellite_alignment_hybridization_p=0.5):
        """
        Parameters
        ==========
        satellite_alignment_strength : float
             parameter between [-1,1] that sets the alignment strength between perfect anti-alignment and perfect alignment

        satellite_alignment_hybridization_p : float
            parameter between [0,1] that sets the hyrbid alignment vector between the radial and major axis.

        Notes
        =====
        If the kwargs or table contain a key "satellite_alignment_strength", this will be used instead.
        """

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['inherit_halocat_properties', 'assign_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = ['halo_x', 'halo_y', 'halo_z',
            'halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z', 'halo_rvir']

        self._additional_kwargs_dict = dict(inherit_halocat_properties=['Lbox'])

        self._methods_to_inherit = (
            ['inherit_halocat_properties', 'assign_orientation'])

        self.param_dict = ({
            'satellite_alignment_a1': satellite_alignment_a1,
            'satellite_alignment_a2': satellite_alignment_a2,
            'satellite_alignment_alpha1': satellite_alignment_alpha1,
            'satellite_alignment_alpha2': satellite_alignment_alpha2})

    def inherit_halocat_properties(self, seed=None, **kwargs):
        """
        """
        Lbox = kwargs['Lbox']
        self._Lbox = Lbox

    def assign_orientation(self, **kwargs):
        r"""
        assign a a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis
        """
        if 'table' in kwargs.keys():
            table = kwargs['table']
            halo_x = table['halo_x']
            halo_y = table['halo_y']
            halo_z = table['halo_z']
            Ax = table[self.list_of_haloprops_needed[3]]
            Ay = table[self.list_of_haloprops_needed[4]]
            Az = table[self.list_of_haloprops_needed[5]]
            halo_r = table['halo_rvir']
            Lbox = self._Lbox
        else:
            halo_x = kwargs['halo_x']
            halo_y = kwargs['halo_y']
            halo_z = kwargs['halo_z']
            Ax = kwargs['halo_axisA_x']
            Ay = kwargs['halo_axisA_y']
            Az = kwargs['halo_axisA_z']
            halo_r = kwargs['halo_rvir']
            Lbox = kwargs['Lbox']

        Ngal = len(Ax)

        # define halo-center - satellite vector
        dx = (x - halo_x)
        mask = dx>Lbox[0]/2.0
        dx[mask] = dx[mask] - Lbox[0]
        mask = dx<-1.0*Lbox[0]/2.0
        dx[mask] = dx[mask] + Lbox[0]

        dy = (y - halo_y)
        mask = dy>Lbox[1]/2.0
        dy[mask] = dy[mask] - Lbox[1]
        mask = dy<-1.0*Lbox[1]/2.0
        dy[mask] = dy[mask] + Lbox[1]

        dz = (z - halo_z)
        mask = dz>Lbox[2]/2.0
        dz[mask] = dz[mask] - Lbox[2]
        mask = dz<-1.0*Lbox[2]/2.0
        dz[mask] = dz[mask] + Lbox[2]

        # radial vector
        v1 = normalized_vectors(np.vstack((dx, dy, dz)).T)

        # major axis orientation
        v2 = normalized_vectors(np.vstack((Ax, Ay, Az)).T)

        # account for handedness by randomly flipping alignment components
        seed = kwargs.get('seed', None)
        with NumpyRNGContext(seed):
             uran1 = np.random.random(Ngal)
        if seed is not None:
            seed = seed + 1
        with NumpyRNGContext(seed):
             uran2 = np.random.random(Ngal)
        flip1 = np.ones(Ngal)
        flip1[uran1 < 0.5] = -1.0
        flip2 = np.ones(Ngal)
        flip2[uran2 < 0.5] = -1.0
        v1 = flip1[:, np.newaxis]*v1
        v2 = flip2[:, np.newaxis]*v2

        # calculate scaled halo virial radius
        r = np.sqrt(dx**2 + dy**2 + dz**2)/halo_r

         # get alignment strength for each galaxy
        if 'table' in kwargs.keys():
            try:
                p = table['satellite_alignment_strength']
            except KeyError:
                msg = ('`satellite_alignment_strength` not detected in the table, using value in self.param_dict.')
                warn(msg)
                p = np.ones(len(table))*self.param_dict['satellite_alignment_strength']
        else:
            N = len(self.param_dict['x'])
            p = np.ones(N*self.param_dict['satellite_alignment_strength'])

        # get major to radial parameter
        a = self.radial_hybrid_alignment_vector_parameter(r)

        # define alignment vector inbetween v1 and v2
        v3 = normalized_vectors(vectors_between_list_of_vectors(v1, v2, a))

        # get galaxy major axis
        major_v = axes_correlated_with_input_vector(v3, p=p)

        # randomly set minor axis orientation
        minor_v = random_perpendicular_directions(major_v)

        # the intermediate axis is determined
        inter_v = vectors_normal_to_planes(major_v, minor_v)

        mask = (table['gal_type'] == self.gal_type)

        # add orientations to the galaxy table
        table['galaxy_axisA_x'][mask] = major_v[mask, 0]
        table['galaxy_axisA_y'][mask] = major_v[mask, 1]
        table['galaxy_axisA_z'][mask] = major_v[mask, 2]

        table['galaxy_axisB_x'][mask] = inter_v[mask, 0]
        table['galaxy_axisB_y'][mask] = inter_v[mask, 1]
        table['galaxy_axisB_z'][mask] = inter_v[mask, 2]

        table['galaxy_axisC_x'][mask] = minor_v[mask, 0]
        table['galaxy_axisC_y'][mask] = minor_v[mask, 1]
        table['galaxy_axisC_z'][mask] = minor_v[mask, 2]

        return table

    def radial_satellite_alignment_strength(self, r):
        """
        strength of alignment as a function of scaled halo-centric radius
        """
        p = power_law(r, self.param_dict['satellite_alignment_a2'], self.param_dict['satellite_alignment_alpha2'])
        p[p>=0.99]=0.99
        return p

    def radial_hybrid_alignment_vector_parameter(self, r):
        """
        hybrid alignment vector parameter
        """
        a = power_law(r, self.param_dict['satellite_alignment_a1'], self.param_dict['satellite_alignment_alpha1'])
        a[a>1.0]=1.0
        return a


def alignment_strength(p):
    r"""
    convert alignment strength argument to shape parameter for costheta distribution
    """

    p = np.atleast_1d(p)
    k = np.zeros(len(p))
    p = p*np.pi/2.0
    k = np.tan(p)
    mask = (p == 1.0)
    k[mask] = np.inf
    mask = (p == -1.0)
    k[mask] = -1.0*np.inf
    return -1.0*k


def inverse_alignment_strength(k):
    r"""
    convert shape parameter for costheta distribution to alignment strength
    """

    k = np.atleast_1d(k)
    p = np.zeros(len(k))

    k = k
    p = -1.0*np.arctan(k)/(np.pi/2.0)

    return p


def axes_correlated_with_z(p, seed=None):
    r"""
    Calculate a list of 3d unit-vectors whose orientation is correlated
    with the z-axis (0, 0, 1).
    Parameters
    ----------
    p : ndarray
        Numpy array with shape (npts, ) defining the strength of the correlation
        between the orientation of the returned vectors and the z-axis.
        Positive (negative) values of `p` produce galaxy principal axes
        that are statistically aligned with the positive (negative) z-axis;
        the strength of this alignment increases with the magnitude of p.
        When p = 0, galaxy axes are randomly oriented.

    seed : int, optional
        Random number seed used to choose a random orthogonal direction

    Returns
    -------
    unit_vectors : ndarray
        Numpy array of shape (npts, 3)

    Notes
    -----
    The `axes_correlated_with_z` function works by modifying the standard method
    for generating random points on the unit sphere. In the standard calculation,
    the z-coordinate :math:`z = \cos(\theta)`, where :math:`\cos(\theta)` is just a
    uniform random variable. In this calculation, :math:`\cos(\theta)` is not
    uniform random, but is instead implemented as a clipped power law
    implemented with `scipy.stats.powerlaw`.
    """

    p = np.atleast_1d(p)
    npts = p.shape[0]

    with NumpyRNGContext(seed):
        phi = np.random.uniform(0, 2*np.pi, npts)
        # sample cosine theta nonuniformily to correlate with in z-axis
        if np.all(p == 0):
            uran = np.random.uniform(0, 1, npts)
            cos_t = uran*2.0 - 1.0
        else:
            k = alignment_strength(p)
            d = DimrothWatson()
            cos_t = d.rvs(k)

    sin_t = np.sqrt((1.-cos_t*cos_t))

    x = sin_t * np.cos(phi)
    y = sin_t * np.sin(phi)
    z = cos_t

    return np.vstack((x, y, z)).T


def axes_correlated_with_input_vector(input_vectors, p=0., seed=None):
    r"""
    Calculate a list of 3d unit-vectors whose orientation is correlated
    with the orientation of `input_vectors`.

    Parameters
    ----------
    input_vectors : ndarray
        Numpy array of shape (npts, 3) storing a list of 3d vectors defining the
        preferred orientation with which the returned vectors will be correlated.
        Note that the normalization of `input_vectors` will be ignored.

    p : ndarray, optional
        Numpy array with shape (npts, ) defining the strength of the correlation
        between the orientation of the returned vectors and the z-axis.
        Default is zero, for no correlation.
        Positive (negative) values of `p` produce galaxy principal axes
        that are statistically aligned with the positive (negative) z-axis;
        the strength of this alignment increases with the magnitude of p.
        When p = 0, galaxy axes are randomly oriented.

    seed : int, optional
        Random number seed used to choose a random orthogonal direction

    Returns
    -------
    unit_vectors : ndarray
        Numpy array of shape (npts, 3)
    """

    input_unit_vectors = normalized_vectors(input_vectors)
    assert input_unit_vectors.shape[1] == 3
    npts = input_unit_vectors.shape[0]

    z_correlated_axes = axes_correlated_with_z(p, seed)

    z_axes = np.tile((0, 0, 1), npts).reshape((npts, 3))

    angles = angles_between_list_of_vectors(z_axes, input_unit_vectors)
    rotation_axes = vectors_normal_to_planes(z_axes, input_unit_vectors)
    matrices = rotation_matrices_from_angles(angles, rotation_axes)

    return rotate_vector_collection(matrices, z_correlated_axes)


