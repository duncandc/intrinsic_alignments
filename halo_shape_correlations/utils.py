"""
functions to facilitate analysis
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

_all__ = ('empirical_cdf', 'binned_bootstrap')

__author__=['Duncan Campbell']

def empirical_cdf(x, bins=None):
    """
    calculate the emprical cumulativr distribution

    Parameters
    ----------
    x : array_like

    bins : array_like
        if bins are passed, the cdf will be returned at the values in `bins`,
        otherwise, the cdf will be returned for every value in `x`

    Returns
    -------
    cdf :  numpy.array
    """

    x = np.sort(x)

    if bins is None:
        cdf = np.ones(len(x))
        cdf = np.cumsum(cdf)/np.sum(cdf)
    else:
        bins = np.sort(bins)
        inds = np.searchsorted(x, bins)
        x = x[inds]
        # pad cdf with zero at beginning
        cdf = np.ones(len(x)+1)
        cdf[0]=0
        cdf = np.cumsum(cdf)/np.sum(cdf)

        cdf = cdf[inds]

    return cdf, x


def binned_bootstrap_1D(x, bins, statistic='mean', Nboot=100):
    """
    calculate the mean value and standard deviation of a statistic calculated in bins by bootstrapping.
    This function is modelled off of scipy.stats.binned_statistic.

    Parameters
    ----------
    x : array_like

    bins : array_like

    statistic : string or callable, optional
        The statistic to compute (default is 'mean').

    Nboot : int

    Returns
    -------
    statistic : numpy.array
        The values of the selected statistic in each bin.

    error: numpy.array
        The bootstrap estimate of the uncertaintiy on the statistic in each bin.

    Examples
    --------
    First, define some random data:
    >>> x = np.random.random(100)

    Define some bins:
    >>> bins = np.linspace(0,1,11)

    Find the mean in each bin and estimate the error on the mean by bootstrap sampling 100 times:
    >>> mean_in_bins, err = binned_bootstrap_1D(x, bins, statistic='mean', Nboot=100)
    """

    edges = np.asarray(bins, float)
    nbin = np.asarray(len(edges) + 1)  # +1 for outlier bins
    dedges = np.diff(edges)

    # Compute the bin number each element in `x` falls into
    bin_number = np.digitize(x, bins)

    # Using `digitize`, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    # Find the rounding precision
    decimal = int(-np.log10(dedges.min())) + 6
    # Find which points are on the rightmost edge.
    on_edge = np.where(np.around(sample[:], decimal) == np.around(edges[-1], decimal))[0]
    # Shift these points one bin to the left.
    bin_number[on_edge] -= 1

    result = np.empty((Nboot, nbin), float)

    # loop through each bin
    for i in range(1, len(bins)):
        mask = (bin_number==i)
        xx = x[mask]
        for j in range(0, Nboot):
            xxx = np.random.choice(xx, size=len(xx), replace=True)
            if statistic == 'mean':
                result[j, i] = np.mean(xxx)
            elif statistic == 'std':
                result[j, i] = np.std(xxx)
            elif statistic == 'count':
                result[j, i] = len(xx)
            elif statistic == 'sum':
                result[j, i] = np.sum(xxx)
            elif statistic == 'median':
                result[j, i] = np.median(xxx)
            elif statistic == 'min':
                result[j, i] = np.min(xxx)
            elif statistic == 'max':
                result[j, i] = np.max(xxx)
            elif callable(statistic):
                result[j, i] = statistic(xxx)
            else:
                msg = ('`statistic` must be a callable function.')
                raise ValueError(msg)

    mean_result = np.mean(result, axis=0)
    err = np.std(resutl, axis=0)

    return mean_result, err


def symmetrize_angular_distribution(theta, radians=True):
    """
    Return theta such that sign[cos(theta)] is equally likley to be 1 and -1.

    parameters
    ----------
    theta : arra_like
        an array of angles between [0.0,np.pi]

    radians :  bool
        boolean indicating if `theta` is in radians.
        If False, it is assummed `theta` is in degrees.

    Returns
    -------
    theta : numpy.array
    """

    if not radians:
        theta = np.radians(theta)

    dtheta = np.fabs(np.pi - np.fabs(theta))

    uran = np.random.random(len(costheta_1))
    result = np.pi + dtheta
    result[uran < 0.5] = -1.0*result[uran < 0.5]

    return result




