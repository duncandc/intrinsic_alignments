"""
convenience functions to calculate jackknife errors
for alignment correlation function mock observables
"""

from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import numpy as np
import time
#from halotools.mock_observables.alignments import ed_3d, ee_3d
#from halotools.mock_observables.alignments import (ee_3d_one_two_halo_decomp,
#                                                   ed_3d_one_two_halo_decomp)
from halotools_ia.correlation_functions import ed_3d, ee_3d
from halotools_ia.correlation_functions import (ee_3d_one_two_halo_decomp,
                                                   ed_3d_one_two_halo_decomp)
from halotools.mock_observables.catalog_analysis_helpers import cuboid_subvolume_labels
from halotools.custom_exceptions import HalotoolsError


__all__ = ('jackknife_ed_3d', 'jackknife_ee_3d',
           'jackknife_ee_3d_one_two_halo_decomp',
           'jackknife_ed_3d_one_two_halo_decomp')

__author__ = ('Duncan Campbell',)


def jackknife_ed_3d(sample1, orientations1, sample2, rbins,
                    Nsub=[5, 5, 5], period=None,
                    num_threads=1, verbose=False,
                    approx_cell1_size=None, approx_cell2_size=None):
    """
    """

    sample1, sample2, j_index_1, j_index_2, Nsub, N_sub_vol, Lbox, PBCs =\
        _process_args(sample1, sample2, period, Nsub)

    # loop through jackknife samples (note that zero is not used as a label)
    results = np.zeros((N_sub_vol+1, len(rbins)-1))
    for i in range(0, N_sub_vol+1):
        start = time.time()

        # remove one sample at a time
        mask1 = (j_index_1 != i)
        mask2 = (j_index_2 != i)

        results[i, :] = ed_3d(sample1[mask1], orientations1[mask1],
                              sample2[mask2], rbins,
                              period=period, num_threads=num_threads,
                              approx_cell1_size=approx_cell1_size,
                              approx_cell2_size=approx_cell2_size)

        dt = time.time()-start
        if (verbose is True) & (i == 0):
            print("estimated time to complete jackknife",
                  "calculation (s): {0}".format(dt*N_sub_vol))

    # get jackknife samples
    result_sub = results[1:, :]

    # get full sample result
    # result = results[0, :]
    result = np.mean(result_sub, axis=0)

    # calculate the covariance matrix
    cov = np.matrix(np.cov(result_sub.T, bias=True))*(N_sub_vol-1.0)

    return result, cov


def jackknife_ee_3d(sample1, orientations1, sample2, orientations2, rbins,
                    Nsub=[5, 5, 5],
                    period=None, num_threads=1, verbose=False,
                    approx_cell1_size=None, approx_cell2_size=None):
    """
    """

    sample1, sample2, j_index_1, j_index_2, Nsub, N_sub_vol, Lbox, PBCs =\
        _process_args(sample1, sample2, period, Nsub)

    # loop through jackknife samples (note that zero is not used as a label)
    results = np.zeros((N_sub_vol+1, len(rbins)-1))
    for i in range(0, N_sub_vol+1):
        start = time.time()

        # remove one sample at a time
        mask1 = (j_index_1 != i)
        mask2 = (j_index_2 != i)

        results[i, :] = ee_3d(sample1[mask1], orientations1[mask1],
                              sample2[mask2], orientations2[mask2], rbins,
                              period=period, num_threads=num_threads,
                              approx_cell1_size=None, approx_cell2_size=None)

        dt = time.time()-start
        if (verbose is True) & (i == 0):
            print("estimated time to complete jackknife",
                  "calculation (s): {0}".format(dt*N_sub_vol))

    # get jackknife samples
    result_sub = results[1:, :]

    # get full sample result
    # result = results[0, :]
    result = np.mean(result_sub, axis=0)

    # calculate the covariance matrix
    cov = np.matrix(np.cov(result_sub.T, bias=True))*(N_sub_vol-1.0)

    return result, cov


def jackknife_ee_3d_one_two_halo_decomp(coords_1, orientations_1, host_ids_1,
                                        coords_2, orientations_2, host_ids_2,
                                        rbins, mask1=None, mask2=None,
                                        period=None, num_threads=1,
                                        Nsub=[5, 5, 5], verbose=True):
    """
    """

    coords_1, coords_2, j_index_1, j_index_2, Nsub, N_sub_vol, Lbox, PBCs =\
        _process_args(coords_1, coords_2, period, Nsub)

    if mask1 is None:
        mask1 = np.array([True]*len(coords_1))
    if mask2 is None:
        mask2 = np.array([True]*len(coords_2))

    # loop through jackknife samples (note that zero is not used as a label)
    result_1h = np.zeros((N_sub_vol+1, len(rbins)-1))
    result_2h = np.zeros((N_sub_vol+1, len(rbins)-1))
    for i in range(0, N_sub_vol+1):
        start = time.time()

        j_mask_1 = (j_index_1 != i)
        j_mask_2 = (j_index_2 != i)

        result_1h[i, :], result_2h[i, :] = ee_3d_one_two_halo_decomp(
            coords_1[j_mask_1],
            orientations_1[j_mask_1],
            host_ids_1[j_mask_1],
            coords_2[j_mask_2],
            orientations_2[j_mask_2],
            host_ids_2[j_mask_2],
            rbins,
            mask1=mask1[j_mask_1],
            mask2=mask2[j_mask_2],
            period=period, num_threads=num_threads)

        dt = time.time()-start
        if (verbose is True) & (i == 0):
            print("estimated time to complete jackknife",
                  "calculation (s): {0}".format(dt*N_sub_vol))

    # get jackknofe samples
    result_1h_sub = result_1h[1:, :]
    result_2h_sub = result_2h[1:, :]

    # get full sample result
    # result_1h = result_1h[0, :]
    # result_2h = result_2h[0, :]
    result_1h = np.mean(result_1h_sub, axis=0)
    result_2h = np.mean(result_2h_sub, axis=0)

    # calculate the covariance matrix
    cov_1h = np.matrix(np.cov(result_1h_sub.T, bias=True))*(N_sub_vol-1.0)
    cov_2h = np.matrix(np.cov(result_2h_sub.T, bias=True))*(N_sub_vol-1.0)

    return result_1h, result_2h, cov_1h, cov_2h


def jackknife_ed_3d_one_two_halo_decomp(coords_1, orientations_1, host_ids_1,
                                        coords_2, host_ids_2,
                                        rbins, mask1=None, mask2=None,
                                        period=None, num_threads=1,
                                        Nsub=[5, 5, 5], verbose=True):
    """
    """

    coords_1, coords_2, j_index_1, j_index_2, Nsub, N_sub_vol, Lbox, PBCs =\
        _process_args(coords_1, coords_2, period, Nsub)

    if mask1 is None:
        mask1 = np.array([True]*len(coords_1))
    if mask2 is None:
        mask2 = np.array([True]*len(coords_2))

    # loop through jackknife samples (note that zero is not used as a label)
    result_1h = np.zeros((N_sub_vol+1, len(rbins)-1))
    result_2h = np.zeros((N_sub_vol+1, len(rbins)-1))
    for i in range(0, N_sub_vol+1):
        start = time.time()

        j_mask_1 = (j_index_1 != i)
        j_mask_2 = (j_index_2 != i)

        result_1h[i, :], result_2h[i, :] = ed_3d_one_two_halo_decomp(
            coords_1[j_mask_1],
            orientations_1[j_mask_1],
            host_ids_1[j_mask_1],
            coords_2[j_mask_2],
            host_ids_2[j_mask_2],
            rbins,
            mask1=mask1[j_mask_1],
            mask2=mask2[j_mask_2],
            period=period, num_threads=num_threads)

        dt = time.time()-start
        if (verbose is True) & (i == 0):
            print("estimated time to complete jackknife",
                  "calculation (s): {0}".format(dt*N_sub_vol))

    # get jackknofe samples
    result_1h_sub = result_1h[1:, :]
    result_2h_sub = result_2h[1:, :]

    # get full sample result
    # result_1h = result_1h[0, :]
    # result_2h = result_2h[0, :]
    result_1h = np.mean(result_1h_sub, axis=0)
    result_2h = np.mean(result_2h_sub, axis=0)

    # calculate the covariance matrix
    cov_1h = np.matrix(np.cov(result_1h_sub.T, bias=True))*(N_sub_vol-1.0)
    cov_2h = np.matrix(np.cov(result_2h_sub.T, bias=True))*(N_sub_vol-1.0)

    return result_1h, result_2h, cov_1h, cov_2h


def _process_args(sample1, sample2, period, Nsub):
    """
    utility function to process common function arguments
    """

    # check to make sure Nsub is reasonable
    Nsub = np.atleast_1d(Nsub)
    if len(Nsub) == 1:
        Nsub = np.array([Nsub[0]]*3)
    try:
        assert np.all(Nsub < np.inf)
        assert np.all(Nsub > 0)
    except AssertionError:
        msg = ("`Nsub` must be a bounded positive number in all dimensions.")
        raise HalotoolsError(msg)

    # check to see if we are using periodic boundary conditions
    if period is None:
        PBCs = False
    else:
        PBCs = True

    # determine minimum box size the data occupies.
    if PBCs is False:
        sample1, sample2, randoms, Lbox = _enclose_in_box(sample1, sample2)
    else:
        Lbox = period

    j_index_1, N_sub_vol = cuboid_subvolume_labels(sample1, Nsub, Lbox)
    j_index_2, N_sub_vol = cuboid_subvolume_labels(sample2, Nsub, Lbox)

    return sample1, sample2, j_index_1, j_index_2, Nsub, N_sub_vol, Lbox, PBCs


def _enclose_in_box(data1, data2):
    """
    build axis aligned box which encloses all points.
    shift points so cube's origin is at 0,0,0.
    """

    x1, y1, z1 = data1[:, 0], data1[:, 1], data1[:, 2]
    x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2]

    xmin = np.min([np.min(x1), np.min(x2)])
    ymin = np.min([np.min(y1), np.min(y2)])
    zmin = np.min([np.min(z1), np.min(z2)])
    xmax = np.max([np.max(x1), np.max(x2)])
    ymax = np.max([np.max(y1), np.max(y2)])
    zmax = np.max([np.max(z1), np.max(z2)])

    xyzmin = np.min([xmin, ymin, zmin])
    xyzmax = np.max([xmax, ymax, zmax])-xyzmin

    x1 = x1 - xyzmin
    y1 = y1 - xyzmin
    z1 = z1 - xyzmin
    x2 = x2 - xyzmin
    y2 = y2 - xyzmin
    z2 = z2 - xyzmin

    Lbox = np.array([xyzmax, xyzmax, xyzmax])

    return np.vstack((x1, y1, z1)).T, np.vstack((x2, y2, z2)).T, Lbox

