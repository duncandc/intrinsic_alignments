from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

from halotools.mock_observables.alignments import ed_3d, ee_3d, ee_3d_one_two_halo_decomp, ed_3d_one_two_halo_decomp
from halotools.mock_observables import cuboid_subvolume_labels


def jackknife_ee_3d_one_two_halo_decomp(coords_1, orientations_1, host_ids_1,
                                        coords_2, orientations_2, host_ids_2,
                                        rbins, period, num_threads, Nsub):
    """
    calculate jackknife covariance matrix for EE correlation function

    Returns 
    =======
    correlation_functions : numpy.array
        Two *len(rbins)-1* length array containing the correlation function :math:`\omega_{1\rm h}(r)`
        and :math:`\omega_{2\rm h}(r)` computed in each of the bins defined by input ``rbins``.

    cov : numpy.ndarray
        Two covariance matrices
    """
    
    labels_1 = cuboid_subvolume_labels(coords_1, Nsub, period)[0]
    labels_2 = cuboid_subvolume_labels(coords_2, Nsub, period)[0]
    
    Njack = np.prod(Nsub)
    result_1h = np.zeros((Njack, len(rbins)-1))
    result_2h = np.zeros((Njack, len(rbins)-1))
    for i in range(1,Njack+1):
        mask_1 = (labels_1 != i)
        mask_2 = (labels_2 != i)

        result_1h[i-1,:], result_2h[i-1,:] = ee_3d_one_two_halo_decomp(coords_1[mask_1], orientations_1[mask_1], host_ids_1[mask_1],
                coords_2[mask_2], orientations_2[mask_2], host_ids_2[mask_2],
                rbins,  period=period, num_threads=num_threads)
    
    cov_1h = np.cov(result_1h.T, bias=True)*(Njack - 1.0)
    cov_2h = np.cov(result_2h.T, bias=True)*(Njack - 1.0)

    return result_1h, result_2h, cov_1h, cov_2h

def jackknife_ed_3d_one_two_halo_decomp(coords_1, orientations_1, host_ids_1,
                                        coords_2, host_ids_2,
                                        rbins, period, num_threads, Nsub):
    """
    calculate jackknife covariance matrix for ED correlation function

    Returns 
    =======
    correlation_functions : numpy.array
        Two *len(rbins)-1* length array containing the correlation function :math:`\omega_{1\rm h}(r)`
        and :math:`\omega_{2\rm h}(r)` computed in each of the bins defined by input ``rbins``.

    cov : numpy.ndarray
        Two covariance matrices
    """
    
    labels_1 = cuboid_subvolume_labels(coords_1, Nsub, period)[0]
    labels_2 = cuboid_subvolume_labels(coords_2, Nsub, period)[0]
    
    Njack = np.prod(Nsub)
    result_1h = np.zeros((Njack, len(rbins)-1))
    result_2h = np.zeros((Njack, len(rbins)-1))
    for i in range(1,Njack+1):
        mask_1 = (labels_1 != i)
        mask_2 = (labels_2 != i)

        result_1h[i-1,:], result_2h[i-1,:] = ed_3d_one_two_halo_decomp(coords_1[mask_1], orientations_1[mask_1], host_ids_1[mask_1],
                coords_2[mask_2], host_ids_2[mask_2],
                rbins,  period=period, num_threads=num_threads)

    cov_1h = np.cov(result_1h.T, bias=True)*(Njack - 1.0)
    cov_2h = np.cov(result_2h.T, bias=True)*(Njack - 1.0)

    return result_1h, result_2h, cov_1h, cov_2h


