"""
functions to facilitate unit vector operations
"""
from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


__all__ = ('principal_axes_from_inertia_tensors', 'inertia_tensors')
__author__ = ('Duncan Campbell')


def principal_axes_from_inertia_tensors(inertia_tensors):
    r"""
    Calculate the principal eigenvector of each of the input inertia tensors.

    Parameters
    ----------
    inertia_tensors : ndarray
        Numpy array of shape (n, ndim, ndim) storing a collection of ndim x ndim symmetric
        positive-definite matrices

    Returns
    -------
    principal_axes : ndarray
        Numpy array of shape (n, ndim) storing a collection of 3d principal eigenvectors

    eigenvalues : ndarray
        Numpy array of shape (n, ) storing the eigenvalue of each principal eigenvector

    Notes
    -----
    """

    ndim = np.shape(inertia_tensors)[-1]

    evals, evecs = np.linalg.eigh(inertia_tensors)
    return evecs[:, :, ndim-1], evals[:, ndim-1]


def inertia_tensors(x, weights=None, reduced=True):
    r"""
    Calculate the n1 inertia tensors for a set of n2 points of dimension ndim.

    Parameters
    ----------
    x :  ndarray
        Numpy array of shape (n1, n2, ndim) storing n1 sets of n2 points
        of dimension ndim.

    weights :  ndarray
        Numpy array of shape (n1, n2) storing n1 sets of n2 weights

    reduced : boolean
        bool indicating whether to calculate the reduced or unreduced inertia tensor
    """

    n1, n2, ndim = np.shape(x)


    if weights is None:
    	weights = np.ones((n1,n2))
    elif np.shape(weights) != (n1,n2):
        msg = ('weights array misty nbe of shape (n1,n2)')
        raise ValueError(msg)

    if reduced:
        r_squared = np.sum(x**2, -1)
        I = np.einsum('...ij,...ik->...jk', x/(r_squared[:,:,np.newaxis]), x*weights[:,:,np.newaxis])
        return I/(np.ones((n1,ndim,ndim))*n2)
    else:
        I = np.einsum('...ij,...ik->...jk', x, x*weights)
        return I/(np.ones((n1,ndim,ndim))*n2)


