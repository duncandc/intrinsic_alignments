"""
script to measure EE and ED for degraded satellite alignments
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table
from astropy.io import ascii
import sys
from intrinsic_alignments.project_settings import PROJECT_DIRECTORY
from halotools.mock_observables.alignments import ed_3d, ee_3d
from halotools.sim_manager import CachedHaloCatalog
from intrinsic_alignments.utils.value_added_halocat import halocat_to_galaxy_table
from intrinsic_alignments.ia_models.ia_model_components import CentralAlignment, SatelliteAlignment
from intrinsic_alignments.ia_models.occupation_models import SubHaloPositions


def calculate_ed_ee(halocat, central_alignment_strength, satellite_alignment_strength, rbins):


    gal_sample_mask = (halocat.halo_table['halo_mpeak']>10**12.0)


    table = halocat.halo_table[gal_sample_mask]

    # satellite position model
    satellite_position_model = SubHaloPositions()
    table = Table(np.copy(satellite_position_model.assign_gal_type(table=table)))
    table = Table(np.copy(satellite_position_model.assign_positions(table=table)))

    # central alignment model
    cen_alignment_model = CentralAlignment(central_alignment_strength=central_alignment_strength)
    table = Table(np.copy(cen_alignment_model.assign_central_orientation(table=table)))

    # satellite alignment model
    sat_alignment_model = SatelliteAlignment(satellite_alignment_strength=satellite_alignment_strength)
    table = Table(np.copy(sat_alignment_model.assign_satellite_orientation(table=table)))

    # calculate observables
    galaxy_coords = np.vstack((table['x'],
                               table['y'],
                               table['z'])).T

    galaxy_orientations = np.vstack((table['galaxy_axisA_x'],
                                     table['galaxy_axisA_y'],
                                     table['galaxy_axisA_z'])).T

    gal_omega = ed_3d(galaxy_coords, galaxy_orientations,
                           galaxy_coords,
                           rbins, period=halocat.Lbox, num_threads=4)

    gal_eta = ee_3d(galaxy_coords, galaxy_orientations,
                         galaxy_coords, galaxy_orientations,
                         rbins, period=halocat.Lbox, num_threads=4)

    return gal_omega, gal_eta


def main():

    simname = 'smdpl_400'
    halocat = CachedHaloCatalog(simname=simname,
                                halo_finder='Rockstar',
                                redshift=0.0, dz_tol=0.001,
                                version_name='custom')

    from intrinsic_alignments.utils.value_added_halocat import halocat_to_galaxy_table
    from intrinsic_alignments.ia_models.ia_model_components import CentralAlignment, SatelliteAlignment
    from intrinsic_alignments.ia_models.occupation_models import SubHaloPositions

    rbins = np.logspace(-1,1.8,29)
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0

    mus = np.linspace(0.99,0.0,10)

    N = 1000
    for i in range(100, N):
        for j in range(1, 10):
            print(i, j)
            ed, ee = calculate_ed_ee(halocat, 0.99, mus[j], rbins=rbins)

            # save measurements
            fpath = fpath = PROJECT_DIRECTORY + 'halo_shape_correlations/data/'
            fname = simname + '_' + '{0:.2f}'.format(0.99) + '_' + '{0:.2f}'.format(mus[j]) + '_ed_ee_'+str(i).zfill(4)+'.dat'
            ascii.write([rbin_centers, ed, ee],
                         fpath+fname,
                         names=['r','ed','ee'],
                         overwrite=True)







if __name__ == '__main__':
    main()
