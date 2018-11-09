"""
functions to facilitate unit vector operations
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from astropy.table import Table
from halotools.utils import crossmatch, elementwise_norm, elementwise_dot, normalized_vectors

__all__ = ('halocat_to_galaxy_table', 'symmetrize_angular_distribution', 'pbc_radial_vector')


__author__ = ('Duncan Campbell', 'Andrew Hearin')


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


def symmetrize_angular_distribution(theta, radians=True):
    """
    Return theta such that sign[cos(theta)] is equally likley to be 1 and -1.

    parameters
    ----------
    theta : arra_like
        an array of angles between [0.0,np.pi]

    radians :  bool
        boolean indicating if `theta` is in radians.
        If False, it is assummed `theta` is in degrees.

    Returns
    -------
    theta : numpy.array
    """

    if not radians:
        theta = np.radians(theta)

    dtheta = np.fabs(np.pi - np.fabs(theta))

    uran = np.random.random(len(theta))
    result = np.pi + dtheta
    result[uran < 0.5] = -1.0*result[uran < 0.5]

    if not radians:
        theta = np.degrees(theta)

    return result


def pbc_radial_vector(coords1, coords2, Lbox=None):
    """
    Calulate the radial vector between 3D points, accounting for perodic boundary conditions (PBCs).

    Paramaters
    ==========
    coords1 : array_like
        array of shape (Npts, 3)

    coords2 : array_like
        array of shape (Npts, 3) defining centers

    Lbox : array_like
        array of shape (3,) indicating the PBCs

    Returns
    =======
    d : numpy.array
        array of shape (Npts, 3) of 3D radial vectors between points in `coords1` and `coords2`
    """
    
    # process Lbox argument
    if Lbox is None:
        Lbox = np.array([np.inf]*3)
    else:
        Lbox = np.atleast_1D(Lbox)

    if len(Lbox)==1:
        Lbox = np.array([Lbox[0]]*3)
    
    # points for which to calculate radial vectors
    x1 = coords1[:,0]
    y1 = coords1[:,1]
    z1 = coords1[:,2]
    
    # coordinate centers
    x2 = coords2[:,0]
    y2 = coords2[:,1]
    z2 = coords2[:,2]

    # account for PBCs
    dx = (x1 - x2)
    mask = dx>Lbox[0]/2.0
    dx[mask] = dx[mask] - Lbox[0]
    mask = dx<-1.0*Lbox[0]/2.0
    dx[mask] = dx[mask] + Lbox[0]

    dy = (y1 - y2)
    mask = dy>Lbox[1]/2.0
    dy[mask] = dy[mask] - Lbox[1]
    mask = dy<-1.0*Lbox[1]/2.0
    dy[mask] = dy[mask] + Lbox[1]

    dz = (z1 - z2)
    mask = dz>Lbox[2]/2.0
    dz[mask] = dz[mask] - Lbox[2]
    mask = dz<-1.0*Lbox[2]/2.0
    dz[mask] = dz[mask] + Lbox[2]

    return np.vstack((dx, dy, dz)).T


