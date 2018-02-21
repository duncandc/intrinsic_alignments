""" Numpy kernels for modeling intrinsic alignments
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from scipy.integrate import quad


from ..ia_model_components import DimrothWatson

def test_DimrothWatson():
    """
    test DimrothWatson class
    """

    d = DimrothWatson()

    # test PDF
    k = 1
    P = quad(d.pdf, -1, 1, args=(k,))[0]
    assert np.isclose(P, 1.0)

    k = 0
    P = quad(d.pdf, -1, 1, args=(k,))[0]
    assert np.isclose(P, 1.0)

    k = -1
    P = quad(d.pdf, -1, 1, args=(k,))[0]
    assert np.isclose(P, 1.0)

    # test CDF
    k = 1
    P1 = d.cdf(-1, k=k)
    P2 = d.cdf(1, k=k)
    assert np.isclose(P1, 0.0) & np.isclose(P2, 1.0)

    k = 0
    P1 = d.cdf(-1, k=k)
    P2 = d.cdf(1, k=k)
    assert np.isclose(P1, 0.0) & np.isclose(P2, 1.0)

    k = -1
    P1 = d.cdf(-1, k=k)
    P2 = d.cdf(1, k=k)
    assert np.isclose(P1, 0.0) & np.isclose(P2, 1.0)

    # test rvs
    N = 1000

    k = -1.0
    random_variates = d.rvs(k, size=N)
    assert np.isclose(np.mean(random_variates), 0.0, atol=0.05)

    k = 0.0
    random_variates = d.rvs(k, size=N)
    assert np.isclose(np.mean(random_variates), 0.0, atol=0.05)

    k = 1.0
    random_variates = d.rvs(k, size=N)
    assert np.isclose(np.mean(random_variates), 0.0, atol=0.05)
