r"""
halotools model components for modelling central and scatellite intrinsic alignments
based on models from the literature, specificalluy those studies in Joachimi + (2013)
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from scipy.optimize import minimize
from rotations import rotate_vector_collection
from rotations.mcrotations import random_perpendicular_directions, random_unit_vectors_3d
from rotations.vector_utilities import (elementwise_dot, elementwise_norm, normalized_vectors,
                                        angles_between_list_of_vectors)
from rotations.rotations3d import (vectors_between_list_of_vectors, vectors_normal_to_planes,
                                   rotation_matrices_from_angles)
from watson_distribution import DimrothWatson
from warnings import warn

from scipy.stats import truncnorm
from scipy.special import iv as modified_bessel
from scipy.interpolate import interp1d


__all__ = ('Bett12SatelliteAlignment',
           'Knebe08SatelliteAlignment',
           'Bett12CentralAlignment',
           'Okumura09CentralAlignment',
           )
__author__ = ('Duncan Campbell',)


class Bett12SatelliteAlignment(object):
    r"""
    radial alignment model for late type satellite galaxies
    based on the central model from Bett + (2012) as used in Joachimi + (2013)
    """

    def __init__(self, **kwargs):
        """
        """

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['inherit_halocat_properties',
                                                   'assign_satellite_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = ['halo_x', 'halo_y', 'halo_z', 'halo_rvir']

        # set default box size.
        if 'Lbox' in kwargs.keys():
            self._Lbox = kwargs['Lbox']
        else:
            self._Lbox = np.inf
        # update Lbox if a halo catalog object is passed.
        self._additional_kwargs_dict = dict(inherit_halocat_properties=['Lbox'])

        self._methods_to_inherit = (['assign_satellite_orientation', 'inherit_halocat_properties'])

        self.set_default_params()


    def inherit_halocat_properties(self, seed=None, **kwargs):
        """
        inherit the box size during mock population
        """
        Lbox = kwargs['Lbox']
        self._Lbox = Lbox

    def set_default_params(self):
        """
        Notes
        """
        self.param_dict = {self.gal_type + '_alingment_scatter': 0.55}

    def misalignment_pdf(self, costheta):
        """
        See Bett (2012) eq. 9

        Parameters
        ----------
        costheta : array_like
             misalingment angle in radians
        """
        sigma = self.param_dict[self.gal_type + '_alingment_scatter']
        
        theta = np.arccos(costheta)
        cos_0 = np.cos(0.0)
        kappa = 1.0/(sigma**2)
        I0 = modified_bessel(0.0, kappa*np.sin(theta)*np.sin(0.0))
        result_1 = (kappa)/(2.0*np.sinh(kappa))*np.exp(kappa*costheta*cos_0)*I0
        result_2 = (kappa)/(2.0*np.sinh(kappa))*np.exp(-1.0*kappa*costheta*cos_0)*I0
        return (result_1 + result_2)/2.0


    def misalignment_rvs(self, size):
        """
        Parameters
        ----------
        size : int

        Notes
        -----
        see Joachimi + (2013) eq. 6
        """
        sigma = self.param_dict[self.gal_type + '_alingment_scatter']
        ran_u = np.random.random(size=size)
        cos_theta = sigma**2*np.log(np.exp(-1.0*sigma**(-2.0))+2.0*np.sinh(sigma**(-2.0))*ran_u)
        
        flip = (np.random.random(size=size)>0.5)
        cos_theta[flip] = -1.0*cos_theta[flip]
        theta = np.arccos(cos_theta)

        mask = (theta>np.pi/2.0)
        theta[mask] = theta[mask]-np.pi

        return theta

    def assign_satellite_orientation(self, **kwargs):
        r"""
        assign a a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis

        Returns
        =======
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
        if np.sum(mask)>0:
            major_input_vectors[mask,0] = np.random.random((np.sum(mask)))
            major_input_vectors[mask,1] = np.random.random((np.sum(mask)))
            major_input_vectors[mask,2] = np.random.random((np.sum(mask)))
            msg = ('{0} galaxies have a radial distance equal to zero (or infinity) from their host. '
                   'These galaxies will be re-assigned random alignment vectors.'.format(int(np.sum(mask))))
            warn(msg)

        # use direction perpendicular to the radial vector
        major_input_vectors = random_perpendicular_directions(major_input_vectors)

        # set prim_gal_axis orientation
        theta_ma = self.misalignment_rvs(size=N)

        # rotate alignment vector by theta_ma
        ran_vecs = random_unit_vectors_3d(N)
        mrot = rotation_matrices_from_angles(theta_ma, ran_vecs)
        A_v = rotate_vector_collection(rotm, major_input_vectors)

        # check for nan vectors
        mask = (~np.isfinite(np.sum(np.prod(A_v, axis=-1))))
        if np.sum(mask)>0:
            A_v[mask,0] = np.random.random((np.sum(mask)))
            A_v[mask,1] = np.random.random((np.sum(mask)))
            A_v[mask,2] = np.random.random((np.sum(mask)))
            msg = ('{0} correlated alignment axis(axes) were not found to be not finite. '
                   'These will be re-assigned random vectors.'.format(int(np.sum(mask))))
            warn(msg)

        # randomly set secondary axis orientation
        B_v = random_perpendicular_directions(A_v)

        # the tertiary axis is determined
        C_v = vectors_normal_to_planes(A_v, B_v)

        # use galaxy minor axis as the orientation axis
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


class Knebe08SatelliteAlignment(object):
    r"""
    radial alignment model for early type satellite galaxies
    from Knebe + (2008), arxiv:0802.1917
    """

    def __init__(self, **kwargs):
        """
        """

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['inherit_halocat_properties',
                                                   'assign_satellite_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = ['halo_x', 'halo_y', 'halo_z', 'halo_rvir']

        self.set_default_params()

        # set default box size.
        if 'Lbox' in kwargs.keys():
            self._Lbox = kwargs['Lbox']
        else:
            self._Lbox = np.inf
        # update Lbox if a halo catalog object is passed.
        self._additional_kwargs_dict = dict(inherit_halocat_properties=['Lbox'])

        self._methods_to_inherit = (['assign_satellite_orientation', 'inherit_halocat_properties'])

    def inherit_halocat_properties(self, seed=None, **kwargs):
        """
        inherit the box size during mock population
        """
        Lbox = kwargs['Lbox']
        self._Lbox = Lbox

    def set_default_params(self):
        """
        Notes
        -----
        Parameter values are averages taken from table 3 in Knebe + (2008).
        These are also the values used in Joachimi + (2013).
        """
        self.param_dict = {self.gal_type + '_alingment_A': 2.64,
                           self.gal_type + '_alingment_B': 0.59}

    def misalignment_pdf(self, costheta):
        """
        Parameters
        ----------
        theta : array_like
             misalingment angle in radians

        Notes
        -----
        See eq. 5 in knebe + (2008)
        There is a typo in this equation popinted out in Joachimi + (2013) eq. 7. 
        """
        A = self.param_dict[self.gal_type + '_alingment_A']
        B = self.param_dict[self.gal_type + '_alingment_B']
        x = costheta
        norm = 0.5
        return norm*(A*x**4+B)/(A/5.0 + B)

    def misalignment_cdf(self, costheta):
        """
        Parameters
        ----------
        theta : array_like
             misalingment angle in radians 
        """
        A = self.param_dict[self.gal_type + '_alingment_A']
        B = self.param_dict[self.gal_type + '_alingment_B']
        x = costheta
        return 0.5*0.2*(A*x**5+A+5*B*x +5*B)/(0.2*A+B)

    def misalignment_rvs(self, size):
        """
        Parameters
        ----------
        size : int

        """
        x = np.linspace(-1,1,1000)
        y = self.misalignment_cdf(x)

        f_yx = interp1d(y,x, kind='linear')
        ran_u = np.random.random(size=size)

        theta = np.arccos(f_yx(ran_u))

        mask = (theta>np.pi/2.0)
        theta[mask] = theta[mask]-np.pi

        return theta

    def assign_satellite_orientation(self, **kwargs):
        r"""
        assign a a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis

        Returns
        =======
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
        if np.sum(mask)>0:
            major_input_vectors[mask,0] = np.random.random((np.sum(mask)))
            major_input_vectors[mask,1] = np.random.random((np.sum(mask)))
            major_input_vectors[mask,2] = np.random.random((np.sum(mask)))
            msg = ('{0} galaxies have a radial distance equal to zero (or infinity) from their host. '
                   'These galaxies will be re-assigned random alignment vectors.'.format(int(np.sum(mask))))
            warn(msg)

        # set prim_gal_axis orientation
        theta_ma = self.misalignment_rvs(size=N)

        # rotate alignment vector by theta_ma
        ran_vecs = random_unit_vectors_3d(N)
        mrot = rotation_matrices_from_angles(theta_ma, ran_vecs)
        A_v = rotate_vector_collection(rotm, major_input_vectors)

        # check for nan vectors
        mask = (~np.isfinite(np.sum(np.prod(A_v, axis=-1))))
        if np.sum(mask)>0:
            A_v[mask,0] = np.random.random((np.sum(mask)))
            A_v[mask,1] = np.random.random((np.sum(mask)))
            A_v[mask,2] = np.random.random((np.sum(mask)))
            msg = ('{0} correlated alignment axis(axes) were not found to be not finite. '
                   'These will be re-assigned random vectors.'.format(int(np.sum(mask))))
            warn(msg)

        # randomly set secondary axis orientation
        B_v = random_perpendicular_directions(A_v)

        # the tertiary axis is determined
        C_v = vectors_normal_to_planes(A_v, B_v)

        # use galaxy major axis as orientation axis
        major_v = A_v
        inter_v = B_v
        minor_v = C_v

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


class Bett12CentralAlignment():
    """
    central galaxy misalignment model for late-type galaixes 
    from Bett (2012), arxiv:1108.3717
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

        self.list_of_haloprops_needed = ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']
        self._methods_to_inherit = ([])
        self.set_default_params()


    def set_default_params(self):
        """
        """
        self.param_dict = {self.gal_type + '_alingment_scatter': 0.55}

    def misalignment_pdf(self, costheta):
        """
        See Bett (2012) eq. 9

        Parameters
        ----------
        costheta : array_like
             misalingment angle in radians
        """
        sigma = self.param_dict[self.gal_type + '_alingment_scatter']
        
        theta = np.arccos(costheta)
        cos_0 = np.cos(0.0)
        kappa = 1.0/(sigma**2)
        I0 = modified_bessel(0.0, kappa*np.sin(theta)*np.sin(0.0))
        result_1 = (kappa)/(2.0*np.sinh(kappa))*np.exp(kappa*costheta*cos_0)*I0
        result_2 = (kappa)/(2.0*np.sinh(kappa))*np.exp(-1.0*kappa*costheta*cos_0)*I0
        return (result_1 + result_2)/2.0


    def misalignment_rvs(self, size):
        """
        Parameters
        ----------
        size : int

        Notes
        -----
        see Joachimi + (2013) eq. 6
        """
        sigma = self.param_dict[self.gal_type + '_alingment_scatter']
        ran_u = np.random.random(size=size)
        cos_theta = sigma**2*np.log(np.exp(-1.0*sigma**(-2.0))+2.0*np.sinh(sigma**(-2.0))*ran_u)
        
        flip = (np.random.random(size=size)>0.5)
        cos_theta[flip] = -1.0*cos_theta[flip]
        theta = np.arccos(cos_theta)

        mask = (theta>np.pi/2.0)
        theta[mask] = theta[mask]-np.pi

        return theta
        
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
            Ax = kwargs[self.list_of_haloprops_needed[0]]
            Ay = kwargs[self.list_of_haloprops_needed[1]]
            Az = kwargs[self.list_of_haloprops_needed[2]]

        # number of haloes
        N = len(Ax)

        # set prim_gal_axis orientation
        major_input_vectors = np.vstack((Ax, Ay, Az)).T
        theta_ma = self.misalignment_rvs(size=N)

        # rotate alignment vector by theta_ma
        ran_vecs = random_unit_vectors_3d(N)
        mrot = rotation_matrices_from_angles(theta_ma, ran_vecs)
        A_v = rotate_vector_collection(rotm, major_input_vectors)

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


class Okumura09CentralAlignment():
    """
    central galaxy misalignment model for early-type galaixes 
    from Okumura, Jing & Li (2009), arxiv:0809.3790
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

        self.list_of_haloprops_needed = ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']
        self._methods_to_inherit = ([])
        self.set_default_params()


    def set_default_params(self):
        """
        """
        self.param_dict = {self.gal_type + '_alingment_scatter': np.radians(35.00)}

    def misalignment_pdf(self, theta):
        """
        Parameters
        ----------
        theta : array_like
             misalingment angle in radians
        """

        theta = np.fabs(theta)

        sigma = self.param_dict[self.gal_type + '_alingment_scatter']
        myclip_a = -np.pi/2.0
        myclip_a = 0
        myclip_b = np.pi/2.0
        a, b = (myclip_a - 0.0)/sigma, (myclip_b - 0.0)/sigma
        return truncnorm.pdf(theta, a=a, b=b, scale=sigma)/2.0

    def misalignment_rvs(self, size):
        """
        Parameters
        ----------
        size : int
        """
        sigma = self.param_dict[self.gal_type + '_alingment_scatter']
        myclip_a = 0.0
        myclip_b = np.pi/2.0
        a, b = (myclip_a - 0.0)/sigma, (myclip_b - 0.0)/sigma
        theta =  truncnorm.rvs(size=size, a=a, b=b, scale=sigma)
        uran = np.random.random(size)
        theta[uran < 0.5] = -1.0*theta[uran < 0.5]
        return theta

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
            Ax = kwargs[self.list_of_haloprops_needed[0]]
            Ay = kwargs[self.list_of_haloprops_needed[1]]
            Az = kwargs[self.list_of_haloprops_needed[2]]

        # number of haloes
        N = len(Ax)

        # set prim_gal_axis orientation
        major_input_vectors = np.vstack((Ax, Ay, Az)).T
        theta_ma = self.misalignment_rvs(size=N)

        # rotate alignment vector by theta_ma
        ran_vecs = random_unit_vectors_3d(N)
        mrot = rotation_matrices_from_angles(theta_ma, ran_vecs)
        A_v = rotate_vector_collection(rotm, major_input_vectors)

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