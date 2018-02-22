r"""
Numpy kernels for modeling intrinsic alignments
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from .utils import random_perpendicular_directions, vectors_normal_to_planes, angles_between_list_of_vectors,\
    rotation_matrices_from_angles, rotate_vector_collection, normalized_vectors
from scipy.stats import rv_continuous
from scipy.special import erf, erfi, erfinv


__all__ = ('CentralAlignment', 'RadialSatelliteAlignment', 'MajorAxisSatelliteAlignment')
__author__ = ('Duncan Campbell', 'Andrew Hearin')


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

        # add orientations to the galaxy table
        table['galaxy_axisA_x'][:] = major_v[:, 0]
        table['galaxy_axisA_y'][:] = major_v[:, 1]
        table['galaxy_axisA_z'][:] = major_v[:, 2]

        table['galaxy_axisB_x'][:] = inter_v[:, 0]
        table['galaxy_axisB_y'][:] = inter_v[:, 1]
        table['galaxy_axisB_z'][:] = inter_v[:, 2]

        table['galaxy_axisC_x'][:] = minor_v[:, 0]
        table['galaxy_axisC_y'][:] = minor_v[:, 1]
        table['galaxy_axisC_z'][:] = minor_v[:, 2]


class RadialSatelliteAlignment(object):
    r"""
    alignment model for satellite galaxies
    """
    def __init__(self, satellite_alignment_stregth=0):

        self.gal_type = 'satellites'
        self._mock_generation_calling_sequence = (['assign_orientation'])

        self._galprop_dtypes_to_allocate = np.dtype(
            [(str('galaxy_axisA_x'), 'f4'), (str('galaxy_axisA_y'), 'f4'), (str('galaxy_axisA_z'), 'f4'),
             (str('galaxy_axisB_x'), 'f4'), (str('galaxy_axisB_y'), 'f4'), (str('galaxy_axisB_z'), 'f4'),
             (str('galaxy_axisC_x'), 'f4'), (str('galaxy_axisC_y'), 'f4'), (str('galaxy_axisC_z'), 'f4')])

        self.list_of_haloprops_needed = ['halo_x', 'halo_y', 'halo_z']

        self._methods_to_inherit = (
            ['assign_orientation'])
        self.param_dict = ({
            'satellite_alignment_strenth': satellite_alignment_stregth})

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
        else:
            halo_x = kwargs['halo_x']
            halo_y = kwargs['halo_z']
            halo_z = kwargs['halo_y']

        p = np.ones(len(halo_x))*self.param_dict['satellite_alignment_strenth']

        # define halo-center - satellite vector
        dx = (table['x'] - halo_x)
        dy = (table['y'] - halo_y)
        dz = (table['z'] - halo_z)

        major_input_vectors = np.vstack((dx, dy, dz)).T

        # set major axis orientation
        major_v = axes_correlated_with_input_vector(major_input_vectors, p=p)

        # randomly set minor axis orientation
        minor_v = random_perpendicular_directions(major_v)

        # the intermediate axis is determined
        inter_v = vectors_normal_to_planes(major_v, minor_v)

        # add orientations to the galaxy table
        table['galaxy_axisA_x'][:] = major_v[:, 0]
        table['galaxy_axisA_y'][:] = major_v[:, 1]
        table['galaxy_axisA_z'][:] = major_v[:, 2]

        table['galaxy_axisB_x'][:] = inter_v[:, 0]
        table['galaxy_axisB_y'][:] = inter_v[:, 1]
        table['galaxy_axisB_z'][:] = inter_v[:, 2]

        table['galaxy_axisC_x'][:] = minor_v[:, 0]
        table['galaxy_axisC_y'][:] = minor_v[:, 1]
        table['galaxy_axisC_z'][:] = minor_v[:, 2]


class MajorAxisSatelliteAlignment(object):
    r"""
    alignment model for satellite galaxies
    """
    def __init__(self, satellite_alignment_stregth=0):

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
            'satellite_alignment_strenth': satellite_alignment_stregth})

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

        # add orientations to the galaxy table
        table['galaxy_axisA_x'][:] = major_v[:, 0]
        table['galaxy_axisA_y'][:] = major_v[:, 1]
        table['galaxy_axisA_z'][:] = major_v[:, 2]

        table['galaxy_axisB_x'][:] = inter_v[:, 0]
        table['galaxy_axisB_y'][:] = inter_v[:, 1]
        table['galaxy_axisB_z'][:] = inter_v[:, 2]

        table['galaxy_axisC_x'][:] = minor_v[:, 0]
        table['galaxy_axisC_y'][:] = minor_v[:, 1]
        table['galaxy_axisC_z'][:] = minor_v[:, 2]


class HybridSatelliteAlignment(object):
    r"""
    alignment model for satellite galaxies
    """
    def __init__(self, satellite_alignment_stregth=0, radial_to_major=0.5):

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
            'satellite_alignment_strenth': satellite_alignment_stregth,
            'radial_to_major': radial_to_major})

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
        a = np.ones(len(Ax))*self.param_dict['radial_to_major']

        # define halo-center - satellite vector
        dx = (table['x'] - halo_x)
        dy = (table['y'] - halo_y)
        dz = (table['z'] - halo_z)
        v1 = normalized_vectors(np.vstack((dx, dy, dz)).T)

        # set major axis orientation
        v2 = normalized_vectors(np.vstack((Ax, Ay, Az)).T)

        v3 = a*v1+(1.0-a)*v2
        v3 = normalized_vectors(v3)

        major_v = axes_correlated_with_input_vector(v3, p=p)

        # randomly set minor axis orientation
        minor_v = random_perpendicular_directions(major_v)

        # the intermediate axis is determined
        inter_v = vectors_normal_to_planes(major_v, minor_v)

        # add orientations to the galaxy table
        table['galaxy_axisA_x'][:] = major_v[:, 0]
        table['galaxy_axisA_y'][:] = major_v[:, 1]
        table['galaxy_axisA_z'][:] = major_v[:, 2]

        table['galaxy_axisB_x'][:] = inter_v[:, 0]
        table['galaxy_axisB_y'][:] = inter_v[:, 1]
        table['galaxy_axisB_z'][:] = inter_v[:, 2]

        table['galaxy_axisC_x'][:] = minor_v[:, 0]
        table['galaxy_axisC_y'][:] = minor_v[:, 1]
        table['galaxy_axisC_z'][:] = minor_v[:, 2]


class DimrothWatson(rv_continuous):
    r"""
    distribution of :math:`\cos(\theta)' for a  Dimroth-Watson distribution
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
        caclulate normalization constant
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
        """
        norm = self._norm(k)
        p = norm*np.exp(-1.0*k*x**2)
        return p

    def _cdf(self, x, k):
        r"""
        cumulative distribution function
        """

        # mask for positive and negative k cases
        negative_k = (k < 0) & (k != 0)
        non_zero_k = (k != 0)

        norm = self._norm(k)

        k = np.fabs(k)

        # array to store result
        result = np.zeros(len(k))
        result[non_zero_k] = np.sqrt(np.pi)*(erf(x[non_zero_k]*np.sqrt(k[non_zero_k]))+erf(np.sqrt(k[non_zero_k])))/(4*np.sqrt(k[non_zero_k]))
        result[negative_k] = np.sqrt(np.pi)*(erfi(x[negative_k]*np.sqrt(k[negative_k]))+erfi(np.sqrt(k[negative_k])))/(4*np.sqrt(k[negative_k]))

        return np.where(k == 0, 0.5*x+0.5, 2.0*norm*result)

    def _rvs(self, k):
        r"""
        random variates
        """

        k = np.atleast_1d(k)
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

        result = np.zeros(size)

        # take care of k=0 case
        zero_k = (k == 0)
        uran0 = np.random.random(np.sum(zero_k))*2 - 1.0
        result[zero_k] = uran0

        # apply rejection sampling technique to sample from pdf
        n_sucess = np.sum(zero_k)  # number of sucesessful draws from pdf
        n_remaining = size - np.sum(zero_k)  # remaining draws necessary
        n_iter = 0  # number of sample-rejhect iterations
        kk = k[~zero_k]  # store subset of k values that still need to be sampled
        mask = np.array([False]*size)  # mask indicating which k values have a sucessful sample
        mask[zero_k] = True
        while n_sucess < size:
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

            #store y values
            result[~mask] = y

            # update mask indicating which values need to be redrawn
            mask[~mask] = keep

            # get subset of k values which need to be sampled.
            kk = kk[~keep]

            n_iter += 1
            n_remaining = np.sum(~keep)

        return result

    def g1(self, x, k):
        r"""
        an upper envelope function for k>0
        """
        k = -1*k
        eta = np.sqrt(-1*k)
        C = eta/(np.arctan(eta))
        return (C/(1+eta**2*x**2))

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
        an upper envelope function for k>0
        """
        k = -1*k
        eta = np.sqrt(-1*k)
        return (1.0/eta)*(np.tan(y*np.arctan(eta)))

    def m1(self, k):
        r"""
        eneveloping factor for proposal distribution for pdf for k>0
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


