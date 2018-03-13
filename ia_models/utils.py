"""
functions to facilitate unit vector operations
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from astropy.table import Table
from halotools.utils import crossmatch

__all__ = ('elementwise_dot', 'elementwise_norm', 'normalized_vectors', 'random_perpendicular_directions',
           'rotation_matrices_from_angles', 'rotation_matrices_from_vectors', 'angles_between_list_of_vectors',
           'vectors_normal_to_planes', 'rotate_vector_collection', 'project_onto_plane')


__author__ = ('Andrew Hearin', )


def halocat_to_galaxy_table(halocat):
    """
    convet a Halotools halocat.halo_table and create a test galaxy_table object to test model componenets
    """

    halo_id = halocat.halo_table['halo_id']
    halo_upid = halocat.halo_table['halo_upid']
    host_id = halocat.halo_table['halo_hostid']

    table = Table([halo_id, halo_upid, host_id], names=('halo_id', 'halo_upid', 'halo_hostid'))
    table['x'] = halocat.halo_table['halo_x']
    table['y'] = halocat.halo_table['halo_y']
    table['z'] = halocat.halo_table['halo_z']
    table['vx'] = halocat.halo_table['halo_vx']
    table['vy'] = halocat.halo_table['halo_vy']
    table['vz'] = halocat.halo_table['halo_vz']
    table['halo_mpeak'] = halocat.halo_table['halo_mpeak']

    table['galaxy_axisA_x'] = 0.0
    table['galaxy_axisA_y'] = 0.0
    table['galaxy_axisA_z'] = 0.0

    table['galaxy_axisB_x'] = 0.0
    table['galaxy_axisB_y'] = 0.0
    table['galaxy_axisB_z'] = 0.0

    table['galaxy_axisC_x'] = 0.0
    table['galaxy_axisC_y'] = 0.0
    table['galaxy_axisC_z'] = 0.0

    # centrals vs satellites
    hosts = (halocat.halo_table['halo_upid'] == -1)
    subs = (halocat.halo_table['halo_upid'] != -1)
    table['gal_type'] = 'satellites'
    table['gal_type'][hosts] = 'centrals'
    table['gal_type'][subs] = 'satellites'

    # host halo properties
    inds1, inds2 = crossmatch(halocat.halo_table['halo_hostid'], halocat.halo_table['halo_id'])
    table['halo_x'] = 0.0
    table['halo_x'][inds1] = halocat.halo_table['halo_x'][inds2]
    table['halo_y'] = 0.0
    table['halo_y'][inds1] = halocat.halo_table['halo_y'][inds2]
    table['halo_z'] = 0.0
    table['halo_z'][inds1] = halocat.halo_table['halo_z'][inds2]
    table['halo_mvir'] = 0.0
    table['halo_mvir'][inds1] = halocat.halo_table['halo_mvir'][inds2]
    table['halo_rvir'] = 0.0
    table['halo_rvir'][inds1] = halocat.halo_table['halo_rvir'][inds2]

    table['halo_axisA_x'] = 0.0
    table['halo_axisA_x'][inds1] = halocat.halo_table['halo_axisA_x'][inds2]
    table['halo_axisA_y'] = 0.0
    table['halo_axisA_y'][inds1] = halocat.halo_table['halo_axisA_y'][inds2]
    table['halo_axisA_z'] = 0.0
    table['halo_axisA_z'][inds1] = halocat.halo_table['halo_axisA_z'][inds2]

    return table


def elementwise_dot(x, y):
    r"""
    Calculate the dot product between
    each pair of elements in two input lists of 3d points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    y : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, ) storing the dot product between each
        pair of corresponding points in x and y.
    """
    x = np.atleast_2d(x)
    y = np.atleast_2d(y)
    return np.sum(x*y, axis=1)


def elementwise_norm(x):
    r"""
    Calculate the normalization of each element in a list of 3d points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, ) storing the norm of each 3d point in x.
    """
    x = np.atleast_2d(x)
    return np.sqrt(np.sum(x**2, axis=1))


def normalized_vectors(vectors):
    r"""
    Return a unit-vector for each 3d vector in the input list of 3d points.

    Parameters
    ----------
    x : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d points

    Returns
    -------
    normed_x : ndarray
        Numpy array of shape (npts, 3)

    """
    vectors = np.atleast_2d(vectors)
    npts = vectors.shape[0]
    return vectors/elementwise_norm(vectors).reshape((npts, -1))


def random_perpendicular_directions(v, seed=None):
    r"""
    Given an input list of 3d vectors, v, return a list of 3d vectors
    such that each returned vector has unit-length and is
    orthogonal to the corresponding vector in v.

    Parameters
    ----------
    v : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

    seed : int, optional
        Random number seed used to choose a random orthogonal direction

    Returns
    -------
    result : ndarray
        Numpy array of shape (npts, 3)

    """
    v = np.atleast_2d(v)
    npts = v.shape[0]
    with NumpyRNGContext(seed):
        w = np.random.random((npts, 3))

    vnorms = elementwise_norm(v).reshape((npts, 1))
    wnorms = elementwise_norm(w).reshape((npts, 1))

    e_v = v/vnorms
    e_w = w/wnorms

    v_dot_w = elementwise_dot(e_v, e_w).reshape((npts, 1))

    e_v_perp = e_w - v_dot_w*e_v
    e_v_perp_norm = elementwise_norm(e_v_perp).reshape((npts, 1))
    return e_v_perp/e_v_perp_norm


def rotation_matrices_from_angles(angles, directions):
    r"""
    Calculate a collection of rotation matrices defined by
    an input collection of rotation angles and rotation axes.

    Parameters
    ----------
    angles : ndarray
        Numpy array of shape (npts, ) storing a collection of rotation angles

    directions : ndarray
        Numpy array of shape (npts, 3) storing a collection of rotation axes in 3d

    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of rotation matrices

    Notes
    -----
    The function `rotate_vector_collection` can be used to efficiently
    apply the returned collection of matrices to a collection of 3d vectors
    """
    directions = np.atleast_2d(directions)
    angles = np.atleast_1d(angles)
    npts = directions.shape[0]

    _dnorm = np.sqrt(np.sum(directions*directions, axis=1))
    directions = directions/_dnorm.reshape((npts, 1))

    sina = np.sin(angles)
    cosa = np.cos(angles)

    R1 = np.zeros((npts, 3, 3))
    R1[:, 0, 0] = cosa
    R1[:, 1, 1] = cosa
    R1[:, 2, 2] = cosa

    R2 = directions[..., None] * directions[:, None, :]
    R2 = R2*np.repeat(1.-cosa, 9).reshape((npts, 3, 3))

    directions *= sina.reshape((npts, 1))
    R3 = np.zeros((npts, 3, 3))
    R3[:, [1, 2, 0], [2, 0, 1]] -= directions
    R3[:, [2, 0, 1], [1, 2, 0]] += directions

    return R1 + R2 + R3


def rotation_matrices_from_vectors(v0, v1):
    r"""
    Calculate a collection of rotation matrices defined by the unique
    transformation rotating v1 into v2 about the mutually perpendicular axis.

    Parameters
    ----------
    v0 : ndarray
        Numpy array of shape (npts, 3) storing a collection of initial vector orientations.

        Note that the normalization of `v0` will be ignored.

    v1 : ndarray
        Numpy array of shape (npts, 3) storing a collection of final vectors.

        Note that the normalization of `v1` will be ignored.

    Returns
    -------
    matrices : ndarray
        Numpy array of shape (npts, 3, 3) rotating each v0 into the corresponding v1

    Notes
    -----
    The function `rotate_vector_collection` can be used to efficiently
    apply the returned collection of matrices to a collection of 3d vectors
    """
    v0 = normalized_vectors(v0)
    v1 = normalized_vectors(v1)
    directions = vectors_normal_to_planes(v0, v1)
    angles = angles_between_list_of_vectors(v0, v1)

    return rotation_matrices_from_angles(angles, directions)


def angles_between_list_of_vectors(v0, v1, tol=1e-3):
    r"""
    Calculate the angle between a collection of 3d vectors

    Examples
    --------
    v0 : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `v0` will be ignored.

    v1 : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `v1` will be ignored.

    tol : float, optional
        Acceptable numerical error for errors in angle.
        This variable is only used to round off numerical noise that otherwise
        causes exceptions to be raised by the inverse cosine function.
        Default is 0.001.

    Returns
    -------
    angles : ndarray
        Numpy array of shape (npts, ) storing the angles between each pair of
        corresponding points in v0 and v1.

        Returned values are in units of radians spanning [0, pi].
    """
    v0 = np.atleast_2d(v0)
    v1 = np.atleast_2d(v1)
    npts = v0.shape[0]
    v0 = v0/np.sqrt(np.sum(v0 * v0, axis=1)).reshape((npts, 1))
    v1 = v1/np.sqrt(np.sum(v1 * v1, axis=1)).reshape((npts, 1))

    dot = np.sum(v0 * v1, axis=1)

    #  Protect against tiny numerical excesses beyond the range [-1 ,1]
    mask1 = (dot > 1) & (dot < 1 + tol)
    dot = np.where(mask1, 1., dot)
    mask2 = (dot < -1) & (dot > -1 - tol)
    dot = np.where(mask2, -1., dot)

    return np.arccos(dot)


def vectors_normal_to_planes(x, y):
    r"""
    Given a collection of 3d vectors x and y,
    return a collection of 3d unit-vectors that are orthogonal to x and y.

    Examples
    --------
    x : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `x` will be ignored.

    y : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

        Note that the normalization of `y` will be ignored.

    Returns
    -------
    z : ndarray
        Numpy array of shape (npts, 3). Each 3d vector in z will be orthogonal
        to the corresponding vector in x and y.
    """
    return normalized_vectors(np.cross(x, y))


def rotate_vector_collection(rotation_matrices, vectors):
    r"""
    Given a collection of rotation matrices and a collection of 3d vectors,
    apply each matrix to rotate the corresponding vector.

    Examples
    --------
    rotation_matrices : ndarray
        Numpy array of shape (npts, 3, 3) storing a collection of rotation matrices

    vectors : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors

    Returns
    -------
    rotated_vectors : ndarray
        Numpy array of shape (npts, 3) storing a collection of 3d vectors
    """
    return np.einsum(str('ijk,ik->ij'), rotation_matrices, vectors)


def project_onto_plane(x, n):
    """
    given a collection of 3D vectors x and n,
    project x onto the plane normal to vector n
    """
    n = normalized_vectors(n)
    d = elementwise_dot(x,n)
    return x - d[:,np.newaxis]*n
