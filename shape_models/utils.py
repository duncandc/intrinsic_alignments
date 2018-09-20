r"""
"""

from __future__ import absolute_import, division, print_function, unicode_literals
import numpy as np


__all__=('projected_b_to_a')
__author__=('Duncan Campbell')


def projected_b_to_a(b_to_a, c_to_a, theta, phi):
    """
    Calulate the projected minor to major semi-axis lengths ratios
    for the 2D projectyion of an 3D ellipsodial distribution.

    Parameters
    ----------
    b_to_a : array_like
        array of intermediate axis ratios, b/a

    c_to_a : array_like
        array of minor axis ratios, c/a

    theta : array_like
        orientation angle, where cos(theta) is bounded between :math:`[0,1]`

    phi : array_like
        orientation angle, where phi is bounded between :math:`[0,2\pi]`

    Notes
    -----

    """

    g = c_to_a  # gamma
    e = 1.0 - b_to_a  # ellipticity

    V = (1 - e*(2 - e)*np.sin(phi)**2)*np.cos(theta)**2 + g**2*np.sin(theta)**2

    W = 4*e**2*(2-e)**2*np.cos(theta)**2*np.sin(phi)**2*np.cos(phi)**2

    Z = 1-e*(2-e)*np.cos(phi)**2

    projected_b_to_a = (V+Z-np.sqrt((V-Z)**2+W))/(V+Z+np.sqrt((V-Z)**2+W))

    return projected_b_to_a
