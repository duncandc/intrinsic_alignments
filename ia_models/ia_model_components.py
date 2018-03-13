r"""
halotools model components for modelling central and scatellite alignments
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from intrinsic_alignments.ia_models.utils import random_perpendicular_directions, vectors_normal_to_planes, angles_between_list_of_vectors,\
    rotation_matrices_from_angles, rotate_vector_collection
from halotools.utils import normalized_vectors, vectors_between_list_of_vectors
from scipy.stats import rv_continuous
from scipy.special import erf, erfi, erfinv
from scipy.optimize import minimize

from warnings import warn


__all__ = ('CentralAlignment', 'RadialSatelliteAlignment', 'MajorAxisSatelliteAlignment', 'HybridSatelliteAlignment', 'DimrothWatson')
__author__ = ('Duncan Campbell', 'Andrew Hearin')


def power_law(x, a, alpha):
    """
    power law model
    """
    return a*x**alpha


class CentralAlignment(object):
    r"""
    alignment model for central galaxies
    """
    def __init__(self, central_alignment_stregth=0):
        r"""
        Parameters
        ----------
        alignment_stregth : float
            [-1,1] bounded number indicating alignment strength
        """

        self.gal_type = 'centrals'
        self._mock_generation_calling_sequence = (['assign_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = ['halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']

        self._methods_to_inherit = (
            ['assign_orientation'])
        self.param_dict = ({
            'central_alignment_strenth': central_alignment_stregth})

    def assign_orientation(self, **kwargs):
        r"""
        assign a a set of three orthoganl unit vectors indicating the orientation
        of the galaxies' major, intermediate, and minor axis

        Parameters
        ----------
        halo_axisA_x, halo_axisA_y, halo_axisA_z :  array_like
             x,y,z components of halo major axis
        """
        if 'table' in kwargs.keys():
            table = kwargs['table']
            Ax = table[self.list_of_haloprops_needed[0]]
            Ay = table[self.list_of_haloprops_needed[1]]
            Az = table[self.list_of_haloprops_needed[2]]
        else:
            Ax = kwargs['halo_axisA_x']
            Ay = kwargs['halo_axisA_z']
            Az = kwargs['halo_axisA_y']

        p = np.ones(len(Ax))*self.param_dict['central_alignment_strenth']

        # set major axis orientation
        major_input_vectors = np.vstack((Ax, Ay, Az)).T
        major_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

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


class RadialSatelliteAlignment(object):
    r"""
    alignment model for satellite galaxies
    """

    def __init__(self, satellite_alignment_a1=1.0,  satellite_alignment_alpha1=0.0):

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['inherit_halocat_properties', 'assign_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = ['halo_x', 'halo_y', 'halo_z', 'halo_rvir']

        self._additional_kwargs_dict = dict(inherit_halocat_properties=['Lbox'])

        self._methods_to_inherit = (
            ['assign_orientation', 'inherit_halocat_properties'])
        self.param_dict = ({
            'satellite_alignment_a1': satellite_alignment_a1,
            'satellite_alignment_alpha1': satellite_alignment_alpha1})

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
            halo_r = table['halo_rvir']
            Lbox = self._Lbox
        else:
            halo_x = kwargs['halo_x']
            halo_y = kwargs['halo_z']
            halo_z = kwargs['halo_y']
            halo_r = kwargs['halo_rvir']
            Lbox = kwargs['Lbox']

        #p = np.ones(len(halo_x))*self.param_dict['satellite_alignment_strenth']

        # define halo-center - satellite vector
        dx = (table['x'] - halo_x)
        mask = dx>Lbox[0]/2.0
        dx[mask] = dx[mask] - Lbox[0]
        mask = dx<-1.0*Lbox[0]/2.0
        dx[mask] = dx[mask] + Lbox[0]

        dy = (table['y'] - halo_y)
        mask = dy>Lbox[1]/2.0
        dx[mask] = dy[mask] - Lbox[1]
        mask = dy<-1.0*Lbox[1]/2.0
        dx[mask] = dy[mask] + Lbox[1]

        dz = (table['z'] - halo_z)
        mask = dz>Lbox[2]/2.0
        dx[mask] = dz[mask] - Lbox[2]
        mask = dz<-1.0*Lbox[2]/2.0
        dx[mask] = dz[mask] + Lbox[2]

        # calculate scaled halo virial radius
        r = np.sqrt(dx**2 + dy**2 + dz**2)/halo_r

        major_input_vectors = np.vstack((dx, dy, dz)).T

        # set major axis orientation
        p = self.radial_satellite_alignment_strength(r)
        major_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

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
        p = power_law(r, self.param_dict['satellite_alignment_a1'], self.param_dict['satellite_alignment_alpha1'])
        p[p>=0.99]=0.99
        return p


class MajorAxisSatelliteAlignment(object):
    r"""
    alignment model for satellite galaxies
    """
    def __init__(self, satellite_alignment_a1=1.0,  satellite_alignment_alpha1=0.0):

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['assign_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = ['halo_x', 'halo_y', 'halo_z', 'halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']

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
            halo_y = kwargs['halo_z']
            halo_z = kwargs['halo_y']
            Ax = kwargs['halo_axisA_x']
            Ay = kwargs['halo_axisA_z']
            Az = kwargs['halo_axisA_y']

        p = np.ones(len(Ax))*self.param_dict['satellite_alignment_strenth']

        # set major axis orientation
        major_input_vectors = np.vstack((Ax, Ay, Az)).T
        major_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

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


class HybridSatelliteAlignment(object):
    r"""
    alignment model for satellite galaxies
    """
    def __init__(self, satellite_alignment_a1=0.2, satellite_alignment_a2=0.8,
        satellite_alignment_alpha1=-0.3, satellite_alignment_alpha2=-0.05):
        """
        Parameters
        ==========
        satellite_alignment_a1 : float
            power-law parameter for hybrid alignment vector

        satellite_alignment_a2 : float
            power-law parameter for alignment strength

        satellite_alignment_alpha1 : float
            power-law parameter for hybrid alignment vector

        satellite_alignment_alpha2 : float
            power-law parameter for alignment strength
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
            halo_y = kwargs['halo_z']
            halo_z = kwargs['halo_y']
            Ax = kwargs['halo_axisA_x']
            Ay = kwargs['halo_axisA_z']
            Az = kwargs['halo_axisA_y']
            halo_r = kwargs['halo_rvir']
            Lbox = kwargs['Lbox']

        Ngal = len(Ax)

        # define radial vector
        dx = (table['x'] - halo_x)
        mask = dx > Lbox[0]/2.0
        dx[mask] = dx[mask] - Lbox[0]
        mask = dx < -1.0*Lbox[0]/2.0
        dx[mask] = dx[mask] + Lbox[0]

        dy = (table['y'] - halo_y)
        mask = dy > Lbox[1]/2.0
        dx[mask] = dy[mask] - Lbox[1]
        mask = dy < -1.0*Lbox[1]/2.0
        dx[mask] = dy[mask] + Lbox[1]

        dz = (table['z'] - halo_z)
        mask = dz > Lbox[2]/2.0
        dx[mask] = dz[mask] - Lbox[2]
        mask = dz < -1.0*Lbox[2]/2.0
        dx[mask] = dz[mask] + Lbox[2]

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

        # get alignment radially dependent alignment strength
        p = self.radial_satellite_alignment_strength(r)

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


class DimrothWatson(rv_continuous):
    r"""
    A Dimroth-Watson distribution of :math:`\cos(\theta)'

    Parameters
    ==========
    k : float
        shape paramater

    Notes
    =====
    The Dimroth-Watson distribution is defined as:

    .. math::
        p(\cos(\theta)) = B(k)\exp[-k\cos(\theta)^2]\mathrm{d}\cos(\theta)

    where

    .. math::
        B(k) = \frac{1}{2}int_0^1\exp(-k t^2)\mathrm{d}t

    We assume the ISO convention for spherical coordinates, where :math:`\theta`
    is the polar angle, bounded between :math:`[-\pi, \pi]`, and :math:`\phi`
    is the azimuthal angle, where for a Dimroth-Watson distribution, :math:`phi'
    is a uniform random variable between :math:`[0, 2\pi]`: for all `k`.

    For :math:`k>0`, the distribution of points on a sphere is bipolar.
    For :math:`k=0`, the distribution of points on a sphere is uniform.
    For :math:`k<0`, the distribution of points on a sphere is girdle.

    Note that as :math:`k \rarrow \infty`:

    .. math::
        p(\cos(\theta)) = \frac{1}{2}\left[ \delta(\cos(\theta) + 1) + \delta(\cos(\theta) - 1) \right]\mathrm{d}\cos(\theta)

    and as :math:`k \rarrow -\infty`:

    .. math::
        p(\cos(\theta)) = \frac{1}{2}\delta(\cos(\theta))\mathrm{d}\cos(\theta)

    Needless to say, for large :math:`|k|`, the attributes of this class are approximate and not well tested.
    """

    def _argcheck(self, k):
        r"""
        check arguments
        """
        k = np.asarray(k)
        self.a = -1.0  # lower bound
        self.b = 1.0  # upper bound
        return (k == k)

    def _norm(self, k):
        r"""
        normalization constant
        """

        k = np.atleast_1d(k)

        # mask for positive and negative k cases
        negative_k = (k < 0) & (k != 0)
        non_zero_k = (k != 0)

        # now ignore sign of k
        k = np.fabs(k)

        # array to store result
        norm = np.zeros(len(k))

        # for k>0
        norm[non_zero_k] = 4.0*np.sqrt(np.pi)*erf(np.sqrt(k[non_zero_k]))/(4.0*np.sqrt(k[non_zero_k]))
        # for k<0
        norm[negative_k] = 4.0*np.sqrt(np.pi)*erfi(np.sqrt(k[negative_k]))/(4.0*np.sqrt(k[negative_k]))

        # ignore divide by zero in the where statement
        with np.errstate(divide='ignore', invalid='ignore'):
            return np.where(k == 0, 0.5, 1.0/norm)

    def _pdf(self, x, k):
        r"""
        probability distribution function

        Parameters
        ==========
        k : float
            shape parameter

        See the Notes section for a discussion of large and small `k`.
        """

        # process arguments
        k = np.atleast_1d(k).astype(np.float64)
        x = np.atleast_1d(x).astype(np.float64)

        with np.errstate(over='ignore', invalid='ignore'):
            norm = self._norm(k)
            p = norm*np.exp(-1.0*k*x**2)
            p = np.nan_to_num(p)

        # deal with the edge cases
        epsilon = np.finfo(float).eps
        edge_mask = (p >= 1.0/epsilon) | (p == 0.0)
        p[edge_mask] = 0.0

        # large positive k (bipolar)
        bipolar = (x >= (1.0 - epsilon)) | (x <= (-1.0 + epsilon))
        p[bipolar & edge_mask & (k>1)] = 1.0/(2.0*epsilon)

        # large negative k (girdle)
        girdle = (x >= (0.0 - epsilon)) & (x <= (0.0 + epsilon))
        p[girdle & edge_mask & (k <- 1)] = 1.0/(2.0*epsilon)

        return p

    def _rvs(self, k):
        r"""
        random variate sampling

        Parameters
        ==========
        k : array_like
            array of shape parameters

        Notes
        =====
        The random variate sampling for this distribution is an implementation
        of the rejection-sampling technique.

        The Proposal distributions are taken from Best & Fisher (1986).
        """

        k = np.atleast_1d(k).astype(np.float64)
        size = self._size[0]
        if size != 1:
            if len(k) == size:
                pass
            elif len(k) == 1:
                k = np.ones(size)*k
            else:
                msg = ('if `size` argument is given, len(k) must be 1 or equal to size.')
                raise ValueError(msg)
        else:
            size = len(k)

        # vector to store random variates
        result = np.zeros(size)

        # take care of k=0 case
        zero_k = (k == 0)
        uran0 = np.random.random(np.sum(zero_k))*2 - 1.0
        result[zero_k] = uran0

        # take care of edge cases, i.e. |k| very large
        with np.errstate(over='ignore'):
            x = np.exp(k)
            inf_mask = np.array([False]*size)
        edge_mask = (x == np.inf) | (x == 0.0)
        result[edge_mask & (k>0)] = np.random.choice([1,-1], size=np.sum(edge_mask & (k>0)))
        result[edge_mask & (k<0)] = 0.0

        # apply rejection sampling technique to sample from pdf
        n_sucess = np.sum(zero_k) + np.sum(edge_mask)  # number of sucesessful draws from pdf
        n_remaining = size - n_sucess  # remaining draws necessary
        n_iter = 0  # number of sample-reject iterations
        kk = k[(~zero_k) & (~edge_mask)]  # store subset of k values that still need to be sampled
        mask = np.array([False]*size)  # mask indicating which k values have a sucessful sample
        mask[zero_k] = True

        max_iter = 100
        while (n_sucess < size) & (n_iter < max_iter):
            # get three uniform random numbers
            uran1 = np.random.random(n_remaining)
            uran2 = np.random.random(n_remaining)
            uran3 = np.random.random(n_remaining)

            # masks indicating which envelope function is used
            negative_k = (kk < 0.0)
            positive_k = (kk > 0.0)

            # sample from g(x) to get y
            y = np.zeros(n_remaining)
            y[positive_k] = self.g1_isf(uran1[positive_k], kk[positive_k])
            y[negative_k] = self.g2_isf(uran1[negative_k], kk[negative_k])
            y[uran3 < 0.5] = -1.0*y[uran3 < 0.5]  # account for one-sided isf function

            # calculate M*g(y)
            g_y = np.zeros(n_remaining)
            m = np.zeros(n_remaining)
            g_y[positive_k] = self.g1_pdf(y[positive_k], kk[positive_k])
            g_y[negative_k] = self.g2_pdf(y[negative_k], kk[negative_k])
            m[positive_k] = self.m1(kk[positive_k])
            m[negative_k] = self.m2(kk[negative_k])

            # calulate f(y)
            f_y = self.pdf(y, kk)

            # accept or reject y
            keep = ((f_y/(g_y*m)) > uran2)

            # count the number of succesful samples
            n_sucess += np.sum(keep)

            # store y values
            result[~mask] = y

            # update mask indicating which values need to be redrawn
            mask[~mask] = keep

            # get subset of k values which need to be sampled.
            kk = kk[~keep]

            n_iter += 1
            n_remaining = np.sum(~keep)

        if (n_iter == max_iter):
            msg = ('The maximum number of iterations reached, random variates may not be represnetitive.')
            raise warn(msg)

        return result

    def g1_pdf(self, x, k):
        r"""
        proposal distribution for pdf for k>0
        """
        k = -1*k
        eta = np.sqrt(-1*k)
        C = eta/(np.arctan(eta))
        return (C/(1+eta**2*x**2))/2.0

    def g1_isf(self, y, k):
        r"""
        inverse survival function of proposal distribution for pdf for k>0
        """
        k = -1*k
        eta = np.sqrt(-1*k)
        return (1.0/eta)*(np.tan(y*np.arctan(eta)))

    def m1(self, k):
        r"""
        eneveloping factor for proposal distribution for k>0
        """
        return 2.0*np.ones(len(k))

    def g2_pdf(self, x, k):
        r"""
        proposal distribution for pdf for k<0
        """
        k = -1*k
        norm = 2.0*(np.exp(k)-1)/k
        return (np.exp(k*np.fabs(x)))/norm

    def g2_isf(self, y, k):
        r"""
        inverse survival function of proposal distribution for pdf for k<0
        """
        k = -1.0*k
        C = k/(np.exp(k)-1.0)
        return np.log(k*y/C+1)/k

    def m2(self, k):
        r"""
        eneveloping factor for proposal distribution for pdf for k<0
        """
        k = -1.0*k
        C = k*(np.exp(k)-1)**(-1)
        norm = 2.0*(np.exp(k)-1)/k
        return C*norm

def fit_watson_mixture_model(x, ptol=0.01, max_iter=50, verbose=False):
    """
    fit for the alignment strength of a symmetric dimroth-watson k-componenent mixture model

    Parameters
    ==========
    x : array_like
        A N by k array of cos(theta)

    ptol : float

    max_iter : int
    """

    def f(p, x, r):
        """
        function to minimize in each step
        """
        k = alignment_strenth(p)
        l = watson_mixture_liklihood(x, k=k, f=r)
        return l

    continue_loop=True
    p0 = 0.0
    num_iter = 0
    while continue_loop==True:
        r = watson_mixture_membership(x, p0)
        p1 = minimize(f, (p0), args=(x, r, ), bounds=[(-0.99,0.99)]).x[0]
        num_iter += 1
        dp = (p1-p0)/p1
        if (dp<ptol) | (num_iter>=max_iter):
            continue_loop=False
        if verbose:
            print(num_iter, p1)
        p0=p1

    return p1


def watson_mixture_membership(x, p):
    """
    return the membership ratio for a symmetric dimroth-watson k-componenent mixture model

    Parameters
    ==========
    x : array_like
        A N by k array of cos(theta)

    p : array_like
        probability

    Returns
    =======
    lnL : numpy.array
        log-liklihood sample `x` was drawn from the distribution
    """

    d = DimrothWatson()

    # process arguments
    x = np.atleast_1d(x)

    k = alignment_strenth(p)

    # size of sample
    N = np.shape(x)[0]
    # number of distributions
    N_components = np.shape(x)[1]

    p = np.zeros((N, N_components))
    for i in range(0, N_components):
        p[:,i] = d.pdf(x[:,i], k=k)

    f = np.zeros((N, N_components))
    for i in range(0, N_components):
        f[:,i] = p[:,i]/np.sum(p, axis=-1)

    return f

def watson_mixture_liklihood(x, f, k):
    """
    Return negative log-liklihood of a symmetric dimroth-watson k-componenent mixture model

    Parameters
    ==========
    x : array_like
        A N by k array of cos(theta)

    f: array_like
        membership

    k : float
        shape parameter of the distribution

    Returns
    =======
    lnL : numpy.array
        log-liklihood sample `x` was drawn from the distribution
    """

    # initialize distribution
    d = DimrothWatson()

    # process arguments
    x = np.atleast_1d(x)
    f = np.atleast_1d(f)
    if np.shape(x) != np.shape(f):
        msg = ('`x` and `f` must be the same shape.')
        raise ValueError(msg)

    # size of sample
    N = np.shape(x)[0]
    # number of distributions
    N_components = np.shape(x)[1]

    # calculate the probabilities each point in the sample
    # was drawn from each individual component
    p = np.zeros((N, N_components))
    for i in range(0, N_components):
        p[:,i] = d.pdf(x[:,i], k=k)

    # log-liklihood liklihood
    l = np.zeros((N_components,))
    for i in range(0, N_components):
        l[i] = np.sum(f[:,i]*np.log(p[:,i]))

    return -1.0*np.sum(l)


def erfiinv(y, kmax=100):
    r"""
    inverse imaginary error function for y close to zero, -1 <= y <= 1
    """

    c = np.zeros(kmax)
    c[0] = 1.0
    c[1] = 1.0
    result = 0.0
    for k in range(0, kmax):
        # Calculate C sub k
        if k > 1:
            c[k] = 0.0
            for m in range(0, k):
                term = (c[m]*c[k - 1 - m])/((m + 1.0)*(2.0*m + 1.0))
                c[k] += term
        result += ((-1.0)**k*c[k]/(2.0*k + 1))*((np.sqrt(np.pi)/2)*y)**(2.0*k + 1)
    return result


def alignment_strenth(p):
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


def inverse_alignment_strenth(k):
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
            k = alignment_strenth(p)
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


