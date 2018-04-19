"""
functions to facilitate analysis
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.stats import binned_statistic

_all__ = ('empirical_cdf', 'binned_bootstrap_1d')

__author__=['Duncan Campbell']

def empirical_cdf(x, bins=None):
    """
    calculate the emprical cumulative distribution

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

    cdf = np.ones(len(x))
    cdf = np.cumsum(cdf)/np.sum(cdf)
    if bins is not None:
        bins = np.sort(bins)
        inds = np.searchsorted(x, bins)
        if np.max(inds) == len(x):
            cdf = np.append(cdf, [1.0])
        if np.min(inds) == 0:
            cdf = np.insert(cdf, [0], [0])
        cdf = cdf[inds]

    return cdf, bins


def binned_bootstrap_1d(x, values, bins, statistic='mean', Nboot=100):
    """
    calculate the mean value and standard deviation of a statistic calculated in bins by bootstrapping.
    This function is modelled off of scipy.stats.binned_statistic.

    Parameters
    ----------
    x : array_like
        A sequence of values to be binned.

    values : array_like
        The values on which the statistic will be computed. This must be the same shape as x.

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
    >>> values = np.random.random(100)

    Define some bins:

    >>> bins = np.linspace(0,1,11)

    Find the mean in each bin and estimate the error on the mean by bootstrap sampling 100 times:

    >>> mean_in_bins, err = binned_bootstrap_1D(x, values, bins, statistic='mean', Nboot=100)
    """

    if Nboot==1:
        result = binned_statistic(x, values, bins=bins, statstic=statistic)[0]
        return result, np.zeros(len(result))

    edges = np.asarray(bins, float)
    nbin = len(edges) - 1
    dedges = np.diff(edges)

    # Compute the bin number each element in `x` falls into
    bin_number = np.digitize(x, edges)

    values = np.asarray(values)

    # Using `digitize`, values that fall on an edge are put in the right bin.
    # For the rightmost bin, we want values equal to the right
    # edge to be counted in the last bin, and not as an outlier.
    # Find the rounding precision
    decimal = int(-np.log10(dedges.min())) + 6
    # Find which points are on the rightmost edge.
    on_edge = np.where(np.around(x, decimal) == np.around(edges[-1], decimal))[0]
    # Shift these points one bin to the left.
    bin_number[on_edge] -= 1

    result = np.empty((Nboot, nbin), float)

    # loop through each bootstrap sample
    for i in range(0, Nboot):
        boot_inds = np.random.randint(0, len(x), size=len(x))
        y = values[boot_inds]  # bootstrapped sample of values
        # loop through each bin
        for j in range(0, len(bins)-1):
            mask = (bin_number[boot_inds]==(j+1))
            yy = y[mask]  # bootstrapped sample of xx within the bin
            if statistic == 'mean':
                result[i, j] = np.nan
                result[i, j] = np.mean(yy)
            elif statistic == 'std':
                result[i, j] = 0.0
                result[i, j] = np.std(yy)
            elif statistic == 'count':
                result[i, j] = 0
                result[i, j] = len(yy)
            elif statistic == 'sum':
                result[i, j] = 0.0
                result[i, j] = np.sum(yy)
            elif statistic == 'median':
                result[i, j] = np.nan
                result[i, j] = np.median(yy)
            elif statistic == 'min':
                result[i, j] = np.nan
                result[i, j] = np.min(yy)
            elif statistic == 'max':
                result[i, j] = np.nan
                result[i, j] = np.max(yy)
            elif callable(statistic):
                try:
                    null = statistic([])
                except:
                    null = np.nan
                result[i, j] = null
                result[i, j] = statistic(yy)
            else:
                msg = ('`statistic` must be a callable function.')
                raise ValueError(msg)

    mean_result = np.nanmean(result, axis=0)
    err = np.nanstd(result, axis=0)

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

    if not radians:
        theta = np.degrees(theta)

    return result




