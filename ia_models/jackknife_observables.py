"""
functions to facilitate unit vector operations
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from halotools.mock_observables.alignments import ed_3d, ee_3d
from halotools.mock_observables.catalog_analysis_helpers import cuboid_subvolume_labels
import time


def jackknife_ed_3d(sample1, orientations1, sample2, rbins,
                    Nsub=[5, 5, 5], weights1=None, weights2=None,
                    period=None, num_threads=1, verbose=False,
                    approx_cell1_size=None, approx_cell2_size=None):
    """
    Calculate the  3-D ellipticity-direction correlation function (ED), :math:`\omega(r)`,
    and the covariance matrix, :math:`{C}_{ij}`, between ith and jth radial bin.
    """

    # Process Nsub entry and check for consistency.
    Nsub = np.atleast_1d(Nsub)
    if len(Nsub) == 1:
        Nsub = np.array([Nsub[0]]*3)
    try:
        assert np.all(Nsub < np.inf)
        assert np.all(Nsub > 0)
    except AssertionError:
        msg = "\n Input `Nsub` must be a bounded positive number in all dimensions"
        raise HalotoolsError(msg)
    
    if period is None:
        PBCs = False
    else:
        PBCs = True

    # determine box size the data occupies.
    # This is used in determining jackknife samples.
    if PBCs is False:
        sample1, sample2, randoms, Lbox = _enclose_in_box(sample1, sample2, randoms)
    else:
        Lbox = period

    j_index_1, N_sub_vol = cuboid_subvolume_labels(sample1, Nsub, Lbox)
    j_index_2, N_sub_vol = cuboid_subvolume_labels(sample2, Nsub, Lbox)

    # loop through jackknife samples
    # note that zero is not used as a label,
    # so masking for zero gives the full sample
    results = np.zeros((N_sub_vol+1, len(rbins)-1))
    for i in range(0, N_sub_vol+1):
        start = time.time()

        # remove one sample at a time
        mask1 = (j_index_1 != i)
        mask2 = (j_index_2 != i)

        results[i,:] = ed_3d(sample1[mask1], orientations1[mask1], sample2[mask2], rbins,
                             period=period, num_threads=num_threads,
                             approx_cell1_size=None, approx_cell2_size=None)

        dt = time.time()-start
        if (verbose==True) & (i==0):
            print("estimated time to complete jackknife calculation (s): {0}".format(dt*N_sub_vol) )

    # get full sample result
    result = results[0, :]

    # get jackknofe samples
    result_sub = results[1:, :]

    # calculate the covariance matrix
    cov = np.matrix(np.cov(result_sub.T, bias=True))*(N_sub_vol-1.0)

    return result, cov


def jackknife_ee_3d(sample1, orientations1, sample2, orientations2, rbins,
                    Nsub=[5, 5, 5], weights1=None, weights2=None,
                    period=None, num_threads=1, verbose=False,
                    approx_cell1_size=None, approx_cell2_size=None):
    """
    Calculate the 3-D ellipticity-ellipticity correlation function (ED), :math:`\eta(r)`,
    and the covariance matrix, :math:`{C}_{ij}`, between ith and jth radial bin.
    """

    # Process Nsub entry and check for consistency.
    Nsub = np.atleast_1d(Nsub)
    if len(Nsub) == 1:
        Nsub = np.array([Nsub[0]]*3)
    try:
        assert np.all(Nsub < np.inf)
        assert np.all(Nsub > 0)
    except AssertionError:
        msg = "\n Input `Nsub` must be a bounded positive number in all dimensions"
        raise HalotoolsError(msg)

    if period is None:
        PBCs = False
    else:
        PBCs = True

    # determine box size the data occupies.
    # This is used in determining jackknife samples.
    if PBCs is False:
        sample1, sample2, randoms, Lbox = _enclose_in_box(sample1, sample2, randoms)
    else:
        Lbox = period

    j_index_1, N_sub_vol = cuboid_subvolume_labels(sample1, Nsub, Lbox)
    j_index_2, N_sub_vol = cuboid_subvolume_labels(sample2, Nsub, Lbox)

    # loop through jackknife samples
    # note that zero is not used as a label,
    # so masking for zero gives the full sample
    results = np.zeros((N_sub_vol+1, len(rbins)-1))
    for i in range(0, N_sub_vol+1):
        start = time.time()
        
        # remove one sample at a time
        mask1 = (j_index_1 != i)
        mask2 = (j_index_2 != i)

        results[i,:] = ee_3d(sample1[mask1], orientations1[mask1],
                             sample2[mask2], orientations2[mask2], rbins,
                             period=period, num_threads=num_threads,
                             approx_cell1_size=None, approx_cell2_size=None)

        dt = time.time()-start
        if (verbose==True) & (i==0):
            print("estimated time to complete jackknife calculation (s): {0}".format(dt*N_sub_vol) )

    # get full sample result
    result = results[0, :]

    # get jackknofe samples
    result_sub = results[1:, :]

    # calculate the covariance matrix
    cov = np.matrix(np.cov(result_sub.T, bias=True))*(N_sub_vol-1.0)

    return result, cov


def _enclose_in_box(data1, data2, data3):
    """
    build axis aligned box which encloses all points.
    shift points so cube's origin is at 0,0,0.
    """

    x1, y1, z1 = data1[:, 0], data1[:, 1], data1[:, 2]
    x2, y2, z2 = data2[:, 0], data2[:, 1], data2[:, 2]
    x3, y3, z3 = data3[:, 0], data3[:, 1], data3[:, 2]

    xmin = np.min([np.min(x1), np.min(x2), np.min(x3)])
    ymin = np.min([np.min(y1), np.min(y2), np.min(y3)])
    zmin = np.min([np.min(z1), np.min(z2), np.min(z3)])
    xmax = np.max([np.max(x1), np.max(x2), np.min(x3)])
    ymax = np.max([np.max(y1), np.max(y2), np.min(y3)])
    zmax = np.max([np.max(z1), np.max(z2), np.min(z3)])

    xyzmin = np.min([xmin, ymin, zmin])
    xyzmax = np.max([xmax, ymax, zmax])-xyzmin

    x1 = x1 - xyzmin
    y1 = y1 - xyzmin
    z1 = z1 - xyzmin
    x2 = x2 - xyzmin
    y2 = y2 - xyzmin
    z2 = z2 - xyzmin
    x3 = x3 - xyzmin
    y3 = y3 - xyzmin
    z3 = z3 - xyzmin

    Lbox = np.array([xyzmax, xyzmax, xyzmax])

    return np.vstack((x1, y1, z1)).T,\
        np.vstack((x2, y2, z2)).T,\
        np.vstack((x3, y3, z3)).T, Lbox
