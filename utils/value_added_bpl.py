import numpy as np
from halotools.sim_manager import CachedHaloCatalog
from halotools.utils import crossmatch, normalized_vectors, angles_between_list_of_vectors
from halotools.mock_observables import relative_positions_and_velocities


def load_bpl():
    halocat = CachedHaloCatalog(simname='bolplanck', halo_finder='rockstar',
        redshift=0.0, version_name='halotools_v0p4')
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
    halos['radial_unit_vector'] = r

    # calculate scaled radial distance (r/r_vir)
    scaled_radius = np.zeros(len(halos))
    # ignore divide by zero in this case
    scaled_radius[inds1] = np.divide(radius[inds1], halos['halo_rvir'][inds2],
                                     out=np.zeros_like(radius[inds1]),
                                     where=halos['halo_rvir'][inds2] != 0)
    halos['r_by_rvir'] = radius

    #define major axis of (sub-)haloes
    halos['major_axis'] = normalized_vectors(np.vstack((halos['halo_axisA_x'],
                                               halos['halo_axisA_y'],
                                               halos['halo_axisA_z'])).T)

    #define spin axis of (sub-)haloes
    halos['spin_axis'] = normalized_vectors(np.vstack((halos['halo_jx'],
                                              halos['halo_jy'],
                                              halos['halo_jz'])).T)

    # define host orientation vectors for each sub-halo
    # major axis
    halos['host_major_axis'] = np.copy(halos['major_axis'])
    halos['host_major_axis'][inds1] = halos['major_axis'][inds2]

    # spin axis
    halos['host_spin_axis'] = np.copy(halos['spin_axis'])
    halos['host_spin_axis'][inds1] = halos['spin_axis'][inds2]

    # major axis
    theta_ma_1 = angles_between_list_of_vectors(halos['radial_unit_vector'], halos['major_axis'])
    theta_ma_2 = angles_between_list_of_vectors(halos['host_major_axis'], halos['major_axis'])

    # spin axis
    theta_ma_3 = angles_between_list_of_vectors(halos['radial_unit_vector'], halos['spin_axis'])
    theta_ma_4 = angles_between_list_of_vectors(halos['host_spin_axis'], halos['spin_axis'])

    return halos



