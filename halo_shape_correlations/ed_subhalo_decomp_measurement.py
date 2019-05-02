"""
script to measure 1/2-halo decomposed EE and ED
with jackknife errors for (sub-)haloes
"""

from __future__ import print_function, division
import numpy as np
from astropy.io import ascii
from intrinsic_alignments.project_settings import PROJECT_DIRECTORY
from halotools.sim_manager import CachedHaloCatalog
from intrinsic_alignments.utils.jackknife_observables import (jackknife_ed_3d_one_two_halo_decomp,
                                                              jackknife_ee_3d_one_two_halo_decomp)


def main():

    simname = 'smdpl_400'
    halocat = CachedHaloCatalog(simname=simname,
                                halo_finder='Rockstar',
                                redshift=0.0, dz_tol=0.001,
                                version_name='custom')

    Njack = int(4**3)
    Nside = int(Njack**(1.0/3.0))
    num_threads = 4

    rbins = np.logspace(-1, 1.8, 29)
    rbin_centers = (rbins[:-1]+rbins[1:])/2.0

    halo_coords = np.vstack((halocat.halo_table['halo_x'],
                             halocat.halo_table['halo_y'],
                             halocat.halo_table['halo_z'])).T

    halo_orientations = np.vstack((halocat.halo_table['halo_axisA_x'],
                                   halocat.halo_table['halo_axisA_y'],
                                   halocat.halo_table['halo_axisA_z'])).T

    # define halo sub-samples
    mask = (halocat.halo_table['halo_mpeak'] >= 10**12.0)

    host_ids = halocat.halo_table['halo_hostid']
    centrals = (halocat.halo_table['halo_upid'] == -1)
    satellites = (halocat.halo_table['halo_upid'] != -1)

    # calculate ED correaltion functions
    # all-all
    result1, result2, cov1, cov2 = jackknife_ed_3d_one_two_halo_decomp(halo_coords[mask],
                                        halo_orientations[mask],
                                        host_ids[mask],
                                        halo_coords[mask],
                                        host_ids[mask],
                                        rbins,
                                        mask1=None,
                                        mask2=None,
                                        period=halocat.Lbox,
                                        num_threads=num_threads,
                                        Nsub=Nside)
    err1 = np.sqrt(np.diag(cov1))
    err2 = np.sqrt(np.diag(cov2))

    # central-central
    cc_result1, cc_result2, cc_cov1, cc_cov2 = jackknife_ed_3d_one_two_halo_decomp(halo_coords[mask],
                                        halo_orientations[mask],
                                        host_ids[mask],
                                        halo_coords[mask],
                                        host_ids[mask],
                                        rbins,
                                        mask1=centrals[mask],
                                        mask2=centrals[mask],
                                        period=halocat.Lbox,
                                        num_threads=num_threads,
                                        Nsub=Nside)
    cc_err1 = np.sqrt(np.diag(cc_cov1))
    cc_err2 = np.sqrt(np.diag(cc_cov2))

    # central-satellite
    cs_result1, cs_result2, cs_cov1, cs_cov2 = jackknife_ed_3d_one_two_halo_decomp(halo_coords[mask],
                                        halo_orientations[mask],
                                        host_ids[mask],
                                        halo_coords[mask],
                                        host_ids[mask],
                                        rbins,
                                        mask1=centrals[mask],
                                        mask2=satellites[mask],
                                        period=halocat.Lbox,
                                        num_threads=num_threads,
                                        Nsub=Nside)
    cs_err1 = np.sqrt(np.diag(cs_cov1))
    cs_err2 = np.sqrt(np.diag(cs_cov2))

    # satellite-central
    sc_result1, sc_result2, sc_cov1, sc_cov2 = jackknife_ed_3d_one_two_halo_decomp(halo_coords[mask],
                                        halo_orientations[mask],
                                        host_ids[mask],
                                        halo_coords[mask],
                                        host_ids[mask],
                                        rbins,
                                        mask1=satellites[mask],
                                        mask2=centrals[mask],
                                        period=halocat.Lbox,
                                        num_threads=num_threads,
                                        Nsub=Nside)
    sc_err1 = np.sqrt(np.diag(sc_cov1))
    sc_err2 = np.sqrt(np.diag(sc_cov2))

    # satellite-satellite
    ss_result1, ss_result2, ss_cov1, ss_cov2 = jackknife_ed_3d_one_two_halo_decomp(halo_coords[mask],
                                        halo_orientations[mask],
                                        host_ids[mask],
                                        halo_coords[mask],
                                        host_ids[mask],
                                        rbins,
                                        mask1=satellites[mask],
                                        mask2=satellites[mask],
                                        period=halocat.Lbox,
                                        num_threads=num_threads,
                                        Nsub=Nside)
    ss_err1 = np.sqrt(np.diag(ss_cov1))
    ss_err2 = np.sqrt(np.diag(ss_cov2))

    # save measurements
    fpath = fpath = PROJECT_DIRECTORY + 'halo_shape_correlations/data/'
    fname = simname + '_ed_one_two_halo_decomp.dat'
    ascii.write([rbin_centers,
                 result1, err1,
                 result2, err2,
                 cc_result1, cc_err1,
                 cc_result2, cc_err2,
                 cs_result1, cs_err1,
                 cs_result2, cs_err2,
                 sc_result1, sc_err1,
                 sc_result2, sc_err2,
                 ss_result1, ss_err1,
                 ss_result2, ss_err2
                 ],
                fpath + fname,
                names=['r',
                       'ed_1', 'ed_err_1',
                       'ed_2', 'ed_err_2',
                       'cc_ed_1', 'cc_err_1',
                       'cc_ed_2', 'cc_err_2',
                       'cs_ed_1', 'cs_err_1',
                       'cs_ed_2', 'cs_err_2',
                       'sc_ed_1', 'sc_err_1',
                       'sc_ed_2', 'sc_err_2',
                       'ss_ed_1', 'ss_err_1',
                       'ss_ed_2', 'ss_err_2'
                       ],
                overwrite=True)


if __name__ == '__main__':
    main()
