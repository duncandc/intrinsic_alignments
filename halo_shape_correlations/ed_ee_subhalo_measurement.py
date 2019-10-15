"""
script to measure EE and ED with jackknife errors for (sub-)haloes
"""

from __future__ import print_function, division
import numpy as np
from astropy.io import ascii
from intrinsic_alignments.project_settings import PROJECT_DIRECTORY
from halotools.sim_manager import CachedHaloCatalog
from intrinsic_alignments.utils.jackknife_observables import (jackknife_ed_3d,
                                                              jackknife_ee_3d)


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

    # calculate ED correaltion functions
    halo_omega, halo_omega_cov = jackknife_ed_3d(halo_coords[mask],
                                                 halo_orientations[mask],
                                                 halo_coords[mask],
                                                 rbins,
                                                 Nsub=[Nside, Nside, Nside],
                                                 period=halocat.Lbox,
                                                 num_threads=num_threads,
                                                 verbose=True)
    halo_omega_err = np.sqrt(np.diag(halo_omega_cov))

    # save measurements
    fpath = fpath = PROJECT_DIRECTORY + 'halo_shape_correlations/data/'
    fname = simname + '_ed.dat'
    ascii.write([rbin_centers, halo_omega, halo_omega_err],
                fpath + fname,
                names=['r', 'ed', 'err'],
                overwrite=True)
    fname = simname + '_ed_cov.npy'
    np.save(fpath+fname, halo_omega_cov)

    # calculate EE correaltion functions
    halo_eta, halo_eta_cov = jackknife_ee_3d(halo_coords[mask],
                                             halo_orientations[mask],
                                             halo_coords[mask],
                                             halo_orientations[mask],
                                             rbins,
                                             Nsub=[Nside, Nside, Nside],
                                             period=halocat.Lbox,
                                             num_threads=num_threads,
                                             verbose=True)
    halo_eta_err = np.sqrt(np.diag(halo_eta_cov))

    # save measurements
    fpath = fpath = PROJECT_DIRECTORY + 'halo_shape_correlations/data/'
    fname = simname + '_ee.dat'
    ascii.write([rbin_centers, halo_eta, halo_eta_err],
                fpath+fname,
                names=['r', 'ee', 'err'],
                overwrite=True)
    fname = simname + '_ee_cov.npy'
    np.save(fpath+fname, halo_eta_cov)

if __name__ == '__main__':
    main()
