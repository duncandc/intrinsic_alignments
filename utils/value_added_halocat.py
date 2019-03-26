"""
"""

import numpy as np
from halotools.sim_manager import CachedHaloCatalog
from halotools.utils import crossmatch, normalized_vectors, angles_between_list_of_vectors
from halotools.mock_observables import relative_positions_and_velocities


__all__=['load_value_added_halocat', 'halocat_to_galaxy_table']


def load_value_added_halocat(simname='bolplanck', redshift=0.0, version_name='halotools_v0p4'):
    """
    adds properties to halotools rockstar halo catalogs

    Returns
    -------
    halocat : Halotools halocat object
    """
    halocat = CachedHaloCatalog(simname=simname, halo_finder='rockstar',
                                redshift=redshift, version_name=version_name)
    halos = halocat.halo_table

    inds1, inds2 = crossmatch(halos['halo_hostid'], halos['halo_id'])
    x = halos['halo_x']
    y = halos['halo_x']
    z = halos['halo_z']
    host_x = np.copy(x)
    host_y = np.copy(y)
    host_z = np.copy(z)
    host_x[inds1] = halos['halo_x'][inds2]
    host_y[inds1] = halos['halo_y'][inds2]
    host_z[inds1] = halos['halo_z'][inds2]

    dx = relative_positions_and_velocities(x, host_x, period=halocat.Lbox[0])
    dy = relative_positions_and_velocities(y, host_y, period=halocat.Lbox[1])
    dz = relative_positions_and_velocities(z, host_z, period=halocat.Lbox[2])
    radius = np.sqrt(dx**2+dy**2+dz**2)
    r = normalized_vectors(np.vstack((dx, dy, dz)).T)
    r = np.nan_to_num(r)

    halos['halo_centric_distance'] = radius
    halos['halo_radial_unit_vector'] = r

    # calculate scaled radial distance (r/r_vir)
    scaled_radius = np.zeros(len(halos))
    # ignore divide by zero in this case
    scaled_radius[inds1] = np.divide(radius[inds1], halos['halo_rvir'][inds2],
                                     out=np.zeros_like(radius[inds1]),
                                     where=halos['halo_rvir'][inds2] != 0)
    halos['halo_r_by_rvir'] = radius

    #define major axis of (sub-)haloes
    halos['halo_major_axis'] = normalized_vectors(np.vstack((halos['halo_axisA_x'],
                                                             halos['halo_axisA_y'],
                                                             halos['halo_axisA_z'])).T)

    #define spin axis of (sub-)haloes
    halos['halo_spin_axis'] = normalized_vectors(np.vstack((halos['halo_jx'],
                                                            halos['halo_jy'],
                                                            halos['halo_jz'])).T)

    # define host orientation vectors for each (sub-)halo
    # major axis
    halos['halo_host_major_axis'] = np.copy(halos['halo_major_axis'])
    halos['halo_host_major_axis'][inds1] = halos['halo_major_axis'][inds2]

    # spin axis
    halos['halo_host_spin_axis'] = np.copy(halos['halo_spin_axis'])
    halos['halo_host_spin_axis'][inds1] = halos['halo_spin_axis'][inds2]

    # major axis
    #theta_ma_1 = angles_between_list_of_vectors(halos['halo_radial_unit_vector'], halos['halo_major_axis'])
    #theta_ma_2 = angles_between_list_of_vectors(halos['halo_host_major_axis'], halos['halo_major_axis'])

    # spin axis
    #theta_ma_3 = angles_between_list_of_vectors(halos['halo_radial_unit_vector'], halos['halo_spin_axis'])
    #theta_ma_4 = angles_between_list_of_vectors(halos['halo_host_spin_axis'], halos['halo_spin_axis'])

    halocat.halo_table = halos

    return halocat


def halocat_to_galaxy_table(halocat):
    """
    transform a Halotools halocat.halo_table into a 
    test galaxy_table object, used for testing model componenets
    
    Returns
    -------
    galaxy_table : astropy.table object
    """

    halo_id = halocat.halo_table['halo_id']
    halo_upid = halocat.halo_table['halo_upid']
    host_id = halocat.halo_table['halo_hostid']

    # create galaxy table
    table = Table([halo_id, halo_upid, host_id], names=('halo_id', 'halo_upid', 'halo_hostid'))
    
    # add position information
    table['x'] = halocat.halo_table['halo_x']
    table['y'] = halocat.halo_table['halo_y']
    table['z'] = halocat.halo_table['halo_z']
    table['vx'] = halocat.halo_table['halo_vx']
    table['vy'] = halocat.halo_table['halo_vy']
    table['vz'] = halocat.halo_table['halo_vz']
    
    # add halo mass
    table['halo_mpeak'] = halocat.halo_table['halo_mpeak']

    # add orientation information
    # place holders for now
    table['galaxy_axisA_x'] = 0.0
    table['galaxy_axisA_y'] = 0.0
    table['galaxy_axisA_z'] = 0.0

    table['galaxy_axisB_x'] = 0.0
    table['galaxy_axisB_y'] = 0.0
    table['galaxy_axisB_z'] = 0.0

    table['galaxy_axisC_x'] = 0.0
    table['galaxy_axisC_y'] = 0.0
    table['galaxy_axisC_z'] = 0.0

    # tag centrals vs satellites
    hosts = (halocat.halo_table['halo_upid'] == -1)
    subs = (halocat.halo_table['halo_upid'] != -1)
    table['gal_type'] = 'satellites'
    table['gal_type'][hosts] = 'centrals'
    table['gal_type'][subs] = 'satellites'

    # host halo properties
    inds1, inds2 = crossmatch(halocat.halo_table['halo_hostid'], halocat.halo_table['halo_id'])
    
    # host halo position
    table['halo_x'] = 0.0
    table['halo_x'][inds1] = halocat.halo_table['halo_x'][inds2]
    table['halo_y'] = 0.0
    table['halo_y'][inds1] = halocat.halo_table['halo_y'][inds2]
    table['halo_z'] = 0.0
    table['halo_z'][inds1] = halocat.halo_table['halo_z'][inds2]
    
    # host haloo mass
    table['halo_mvir'] = 0.0
    table['halo_mvir'][inds1] = halocat.halo_table['halo_mvir'][inds2]
    table['halo_rvir'] = 0.0
    table['halo_rvir'][inds1] = halocat.halo_table['halo_rvir'][inds2]

    # assign orientations
    table['halo_axisA_x'] = 0.0
    table['halo_axisA_x'][inds1] = halocat.halo_table['halo_axisA_x'][inds2]
    table['halo_axisA_y'] = 0.0
    table['halo_axisA_y'][inds1] = halocat.halo_table['halo_axisA_y'][inds2]
    table['halo_axisA_z'] = 0.0
    table['halo_axisA_z'][inds1] = halocat.halo_table['halo_axisA_z'][inds2]

    return table


