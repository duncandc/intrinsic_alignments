"""
script to measure galaxy shapes in Illustris
"""

from __future__ import print_function, division
import numpy as np
from astropy.table import Table
from astropy.io import ascii
import sys
from intrinsic_alignments.project_settings import PROJECT_DIRECTORY


def main():

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

    # define galaxy sample
    if sample_name == 'sample_1':
    	m_thresh = 10**10.5
    if sample_name == 'sample_2':
    	m_thresh = 10**10.0
    if sample_name == 'sample_3':
    	m_thresh = 10**9.5

    from Illustris_Shapes.simulation_props import sim_prop_dict
    d = sim_prop_dict[sim_name]
    Lbox = d['Lbox']

    # load vagc and shape catalogs
    fpath = PROJECT_DIRECTORY + 'data/Illustris/'

    # load VAGC
    fname = 'value_added_catalogs/' + sim_name + '_' + str(snapnum) + '_vagc.dat'
    vagc_galaxy_table = Table.read(fpath + fname, format='ascii')

    # load galaxy shape catalog
    fname = 'shape_catalogs/' + sim_name + '_' + str(snapnum) + '_' + shape_type + '_galaxy_shapes.dat'
    galaxy_shape_table = Table.read(fpath + fname, format='ascii')

    # join galaxy shape and vagc tables
    from astropy.table import join
    galaxy_table = join(vagc_galaxy_table, galaxy_shape_table)

    selection_mask = galaxy_table['stellar_mass_all'] >= m_thresh
    print('number of galaxies in selection: ', np.sum(selection_mask))

    gal_coords = np.vstack((galaxy_table['x'],
                    galaxy_table['y'],
                    galaxy_table['z'])).T/1000.0

    # galaxy major_axis
    gal_orientations_major = np.vstack((galaxy_table['av_x'],
                                        galaxy_table['av_y'],
                                        galaxy_table['av_z'])).T


    rbins = np.logspace(-1,1.5,15)
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0

    from halotools.mock_observables import tpcf, tpcf_jackknife

    N=10**5
    print('number of randoms: ', N)
    
    randoms = np.random.random((N,3))*Lbox
    xi, cov = tpcf_jackknife(gal_coords[selection_mask], randoms, rbins, period=Lbox, Nsub=[3, 3, 3])
    err = np.sqrt(np.diag(cov))

    # save measurements
    fpath = fpath = PROJECT_DIRECTORY + 'modelling_illustris/data/'
    fname = sim_name + '_' + str(snapnum) + '-' + sample_name +'_xi.dat'
    ascii.write([rbin_centers, xi, err],
                fpath+fname,
                names=['r','xi','err'],
                overwrite=True)







if __name__ == '__main__':
    main()