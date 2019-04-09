"""
script to measure galaxy shapes in Illustris
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table
from astropy.io import ascii
import sys
from intrinsic_alignments.project_settings import PROJECT_DIRECTORY
import time


def main():

    nu_cens = np.linspace(0,1,2)
    nu_sats = np.linspace(-0.3,0.8, 2)

    # get simulation information
    if len(sys.argv)>1:
        sim_name = sys.argv[1]
        snapnum = int(sys.argv[2])
        shape_type = sys.argv[3]
        sample_name = sys.argv[4]
    else:
        sim_name = 'TNG300-1' # full physics high-res run
        snapnum = 99  # z=0
        shape_type = 'reduced'  # non-reduced, reduced, iterative
        sample_name = 'sample_3'

    # load a test halo catalog
    from halotools.sim_manager import CachedHaloCatalog
    halocat = CachedHaloCatalog(simname='bolplanck', halo_finder='rockstar',
                                redshift=0.0, dz_tol=0.1, version_name='halotools_v0p4')

    from halotools.empirical_models import HodModelFactory

    # define the central occupatoion model
    from halotools.empirical_models import TrivialPhaseSpace, Zheng07Cens
    cens_occ_model =  Zheng07Cens()
    cens_prof_model = TrivialPhaseSpace()

    # define the satellite occupation model
    from halotools.empirical_models import Zheng07Sats
    from halotools.empirical_models import NFWPhaseSpace, SubhaloPhaseSpace
    from intrinsic_alignments.ia_models.anisotropic_nfw_phase_space import AnisotropicNFWPhaseSpace
    sats_occ_model =  Zheng07Sats()
    #sats_prof_model = AnisotropicNFWPhaseSpace()
    sats_prof_model = SubhaloPhaseSpace('satellites', np.logspace(10.5, 15.2, 15))

    # define the alignment models
    from intrinsic_alignments.ia_models.ia_model_components import CentralAlignment,\
        RadialSatelliteAlignment,  MajorAxisSatelliteAlignment, HybridSatelliteAlignment
    central_orientation_model = CentralAlignment()
    satellite_orientation_model = RadialSatelliteAlignment()

    if sample_name == 'sample_1':
        cens_occ_model.param_dict['logMmin'] = 12.54
        cens_occ_model.param_dict['sigma_logM'] = 0.26

        sats_occ_model.param_dict['alpha'] = 1.0
        sats_occ_model.param_dict['logM0'] = 12.68
        sats_occ_model.param_dict['logM1'] = 13.48

        central_orientation_model.param_dict['central_alignment_strength'] = 0.755
        satellite_orientation_model.param_dict['satellite_alignment_strength'] = 0.279
    elif sample_name == 'sample_2':
        cens_occ_model.param_dict['logMmin'] = 11.93
        cens_occ_model.param_dict['sigma_logM'] = 0.26

        sats_occ_model.param_dict['alpha'] = 1.0
        sats_occ_model.param_dict['logM0'] = 12.05
        sats_occ_model.param_dict['logM1'] = 12.85

        central_orientation_model.param_dict['central_alignment_strength'] = 0.64
        satellite_orientation_model.param_dict['satellite_alignment_strength'] = 0.084
    elif sample_name =='sample_3':
        cens_occ_model.param_dict['logMmin'] = 11.61
        cens_occ_model.param_dict['sigma_logM'] = 0.26

        sats_occ_model.param_dict['alpha'] = 1.0
        sats_occ_model.param_dict['logM0'] = 11.8
        sats_occ_model.param_dict['logM1'] = 12.6

        central_orientation_model.param_dict['central_alignment_strength'] = 0.57172919
        satellite_orientation_model.param_dict['satellite_alignment_strength'] = 0.01995

    # combine model components
    model_instance = HodModelFactory(centrals_occupation = cens_occ_model,
                                 centrals_profile = cens_prof_model,
                                 satellites_occupation = sats_occ_model,
                                 satellites_profile = sats_prof_model,
                                 centrals_orientation = central_orientation_model,
                                 satellites_orientation = satellite_orientation_model,
                                 model_feature_calling_sequence = (
                                 'centrals_occupation',
                                 'centrals_profile',
                                 'satellites_occupation',
                                 'satellites_profile',
                                 'centrals_orientation',
                                 'satellites_orientation')
                                )
    
    from intrinsic_alignments.utils.jackknife_observables import jackknife_ed_3d
    from halotools.mock_observables.alignments import ed_3d

    rbins = np.logspace(-1,1.5,15)
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0

    N1 = len(nu_cens)
    N2 = len(nu_sats)
    
    fpath = fpath = PROJECT_DIRECTORY + 'modelling_illustris/data/'
    fname = sim_name + '_' + str(snapnum) + '-' + sample_name +'_model_ed_grid.dat'

    outF = open(fpath + fname, "w")
    
    for i in range(0,N1):
        for j in range(0,N2):

            print(i, j)

            # assign parameters
            central_orientation_model.param_dict['central_alignment_strength'] = nu_cens[i]
            satellite_orientation_model.param_dict['satellite_alignment_strength'] = nu_sats[j]
            
            # populate mock catalog
            start = time.time()
            model_instance.populate_mock(halocat)
            print("time to populate mock: ", time.time() - start)

            mock = model_instance.mock.galaxy_table

            # galaxy coordinates and orientations
            coords = np.vstack((mock['x'],
                                mock['y'],
                                mock['z'])).T

            orientations = np.vstack((mock['galaxy_axisA_x'],
                                      mock['galaxy_axisA_y'],
                                      mock['galaxy_axisA_z'])).T

            # calculate ED
            start = time.time()
            ed = ed_3d(coords, orientations,coords,
                       rbins, period=halocat.Lbox,
                       num_threads=4)
            print("time to calculate ED stat: ", time.time() - start)

            # calculate EE
            start = time.time()
            ee = ee_3d(coords, orientations, coords, orientations,
                       rbins, period=halocat.Lbox,
                       num_threads=4)
            print("time to calculate EE stat: ", time.time() - start)

            s = str(nu_cens[i]) + ' ' + str(nu_sats[j]) + ' ' + np.array_str(ed)[1:-1] + ' ' + np.array_str(ee)[1:-1]
            outF.write(s)
            outF.write("\n")
    
    outF.close()



if __name__ == '__main__':
    main()