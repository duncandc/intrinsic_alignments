r"""
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np

def projected_b_to_a(g, e, theta, phi):
    """
    Parameters
    ----------
    g : array_like
        array of ratios of minor to major axis, :math:`C/A`

    e : array_like
        array of ellipticities, :math:`1-B/A`

    theta : array_like
        orientation angle, where cos(theta) is bounded between :math:`[0,1]`

    phi : array_like
        orientation angle, where phi is bounded between :math:`[0,2\pi]`
    """

    V = (1 - e*(2 - e)*np.sin(phi)**2)*np.cos(theta)**2 + g**2*np.sin(theta)**2

    W = 4*e**2*(2-e)**2*np.cos(theta)**2*np.sin(phi)**2*np.cos(phi)**2

    Z = 1-e*(2-e)*np.cos(phi)**2

    b_to_a = (V+Z-np.sqrt((V-Z)**2+W))/(V+Z+np.sqrt((V-Z)**2+W))

    return b_to_a
