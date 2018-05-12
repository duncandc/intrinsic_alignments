r"""
halotools model components for modelling central and scatellite intrinsic alignments
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
from astropy.utils.misc import NumpyRNGContext
from warnings import warn
from halotools.utils import crossmatch
from halotools.utils import rotation_matrices_from_angles, rotate_vector_collection


__all__ = ()
__author__ = ('Duncan Campbell')

class SubHaloPositions():
    """
	galaxy occupation model that places centrals and satellites in haloes and sub-haloes
	"""

    def __init__(self):
        """

        """
        self._mock_generation_calling_sequence = ['assign_gal_type', 'assign_positions']
        self._galprop_dtypes_to_allocate = np.dtype([(str('gal_type'), 'string'),
                                                     (str('x'), 'f4'), (str('y'), 'f4'), (str('z'), 'f4')])
        self.list_of_haloprops_needed = ['halo_upid', 'halo_hostid', 'halo_id', 'halo_x', 'halo_y', 'halo_z']

    def assign_positions(self, **kwargs):
        """
        assign satellite positions based on subhalo positions.
        """

        
        if 'table' in kwargs.keys():
            table = kwargs['table']
            halo_id = table['halo_id']
            halo_hostid = table['halo_hostid']
            halo_x = table['halo_x']
            halo_y = table['halo_y']
            halo_z = table['halo_z']
        else:
            halo_x = kwargs['halo_x']
            halo_y = kwargs['halo_y']
            halo_z = kwargs['halo_z']
        
        # get satellite positions
        x = halo_x*1.0
        y = halo_y*1.0
        z = halo_z*1.0

        # get host halo positions for satellites
        inds1, inds2 = crossmatch(halo_hostid , halo_id)
        halo_x[inds1] =halo_x[inds2]
        halo_y[inds1] =halo_y[inds2]
        halo_z[inds1] =halo_z[inds2]

        if 'table' in kwargs.keys():
            table['x'] = x*1.0
            table['y'] = y*1.0
            table['z'] = z*1.0
            table['halo_x'] = halo_x*1.0
            table['halo_y'] = halo_y*1.0
            table['halo_z'] = halo_z*1.0
            return table
        else:
            return np.vstack((x,y,z)).T, np.vstack((halo_x,halo_y,halo_z)).T

    def assign_gal_type(self, **kwargs):
        """
        specify central and satellites gal types
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            upid = table['halo_upid']
        else:
            upid = kwargs['halo_upid']


        centrals = (upid == -1)
        satellites = (upid != -1)
        
        if 'table' in kwargs.keys():
            # assign galaxy type
            table['gal_type'] = 'satellites'
            table['gal_type'][centrals] = 'centrals'

            return table
        else:
            result = np.array(['satellites']*len(upid))
            result[centrals] = 'centrals'
            return result


class IsotropicSubhaloPositions():
    """
    galaxy occupation model that places centrals and satellites in haloes and isotropized sub-haloes
	"""

    def __init__(self, **kwargs):

        self._mock_generation_calling_sequence = ['assign_gal_type', 'assign_positions']
        self._galprop_dtypes_to_allocate = np.dtype([(str('gal_type'), 'string'),
                                                     (str('x'), 'f4'), (str('y'), 'f4'), (str('z'), 'f4')])
        self.list_of_haloprops_needed = ['halo_upid', 'halo_x', 'halo_y', 'halo_z', 'halo_hostid']

        # set default box size.
        if 'Lbox' in kwargs.keys():
            self._Lbox = kwargs['Lbox']
        else:
            self._Lbox = np.array([np.inf,np.inf,np.inf])
        # update Lbox if a halo catalog object is passed.
        self._additional_kwargs_dict = dict(inherit_halocat_properties=['Lbox'])

    def inherit_halocat_properties(self, seed=None, **kwargs):
        """
        inherit the box size during mock population
        """
        Lbox = kwargs['Lbox']
        self._Lbox = Lbox

    def assign_positions(self, **kwargs):
        """
        assign satellite positions based on subhalo radial positions and random angular positions.
    	"""

        if 'table' in kwargs.keys():
            table = kwargs['table']
            halo_x = table['halo_x']
            halo_y = table['halo_y']
            halo_z = table['halo_z']
            halo_hostid = table['halo_hostid']
            halo_id = table['halo_id']
            try:
                Lbox = kwargs['Lbox']
            except KeyError:
                Lbox = self._Lbox
        else:
            halo_x = kwargs['halo_x']
            halo_y = kwargs['halo_y']
            halo_z = kwargs['halo_z']
            halo_hostid = kwargs['halo_hostid']
            halo_id = kwargs['halo_id']
            try:
                Lbox = kwargs['Lbox']
            except KeyError:
                Lbox = self._Lbox
        
        # get subhalo positions
        x = halo_x*1.0
        y = halo_y*1.0
        z = halo_z*1.0

        # get host halo positions 
        inds1, inds2 = crossmatch(halo_hostid, halo_id)
        # x-position
        halo_x[inds1] = halo_x[inds2]
        # y-position
        halo_y[inds1] = halo_y[inds2]
        # z-position
        halo_z[inds1] = halo_z[inds2]

        # calculate radial positions
        vec_r, r = radial_distance(x, y, z, halo_x, halo_y, halo_z, Lbox)

        # calculate new positions with same radial distance
        # in the coordinatre systems centered on host haloes
        npts = len(x)
        uran = np.random.uniform(0, 1, npts)
        theta = np.arccos(uran*2.0 - 1.0)
        phi = np.random.uniform(0, 2*np.pi, npts)
        xx = r * np.sin(theta) * np.cos(phi)
        yy = r * np.sin(theta) * np.sin(phi)
        zz = r * np.cos(theta)

        # move back into original cordinate system
        xx = halo_x + xx
        yy = halo_y + yy
        zz = halo_z + zz
        
        # account for PBCs
        mask = (xx < 0.0)
        xx[mask] = xx[mask] + Lbox[0]
        mask = (xx > Lbox[0])
        xx[mask] = xx[mask] - Lbox[0]
        mask = (yy < 0.0)
        yy[mask] = yy[mask] + Lbox[1]
        mask = (yy > Lbox[1])
        yy[mask] = yy[mask] - Lbox[1]
        mask = (zz < 0.0)
        zz[mask] = zz[mask] + Lbox[2]
        mask = (zz > Lbox[2])
        zz[mask] = zz[mask] - Lbox[2]

        if 'table' in kwargs.keys():
            # assign satellite galaxy positions
            try:
                mask = (table['gal_type']=='satellites')
            except KeyError:
                mask = np.array([True]*len(table))
                msg = ("`gal_type` not indicated in `table`.",
                       "The orientation is being assigned for all galaxies in the `table`.")
                print(msg)
            
            table['x'] = halo_x*1.0
            table['y'] = halo_y*1.0
            table['z'] = halo_z*1.0

            table['x'][mask] = xx[mask]
            table['y'][mask] = yy[mask]
            table['z'][mask] = zz[mask]

            table['halo_x'][mask] = halo_x[mask]
            table['halo_y'][mask] = halo_y[mask]
            table['halo_z'][mask] = halo_z[mask]

            return table
        else:
            x = xx
            y = yy
            z = zz
            return np.vstack((x,y,z)).T

    def assign_gal_type(self, **kwargs):
    	"""
        specify central and satellites
    	"""
        
        if 'table' in kwargs.keys():
            table = kwargs['table']
            upid = table['halo_upid']
        else:
            upid = kwargs['halo_upid']


        centrals = (upid == -1)
        satellites = (upid != -1)
        
        if 'table' in kwargs.keys():
            # assign galaxy type
            table['gal_type'] = 'satellites'
            table['gal_type'][centrals] = 'centrals'

            return table
        else:
            result = np.array(['satellites']*len(upid))
            result[centrals] = 'centrals'
            return result


class SemiIsotropicSubhaloPositions():
    """
    galaxy occupation model that places centrals and satellites in haloes and isotropized sub-haloes
    """

    def __init__(self, **kwargs):

        self._mock_generation_calling_sequence = ['assign_gal_type', 'assign_positions']
        self._galprop_dtypes_to_allocate = np.dtype([(str('gal_type'), 'string'),
                                                     (str('x'), 'f4'), (str('y'), 'f4'), (str('z'), 'f4')])
        self.list_of_haloprops_needed = ['halo_upid', 'halo_x', 'halo_y', 'halo_z', 'halo_hostid', 'halo_axisA_x', 'halo_axisA_y', 'halo_axisA_z']

        # set default box size.
        if 'Lbox' in kwargs.keys():
            self._Lbox = kwargs['Lbox']
        else:
            self._Lbox = np.array([np.inf,np.inf,np.inf])
        # update Lbox if a halo catalog object is passed.
        self._additional_kwargs_dict = dict(inherit_halocat_properties=['Lbox'])

    def inherit_halocat_properties(self, seed=None, **kwargs):
        """
        inherit the box size during mock population
        """
        Lbox = kwargs['Lbox']
        self._Lbox = Lbox

    def assign_positions(self, **kwargs):
        """
        assign satellite positions based on subhalo radial positions and random angular positions.
        """

        if 'table' in kwargs.keys():
            table = kwargs['table']
            halo_x = table['halo_x']
            halo_y = table['halo_y']
            halo_z = table['halo_z']
            halo_axisA_x = table['halo_axisA_x']
            halo_axisA_y = table['halo_axisA_y']
            halo_axisA_z = table['halo_axisA_z']
            halo_hostid = table['halo_hostid']
            halo_id = table['halo_id']
            try:
                Lbox = kwargs['Lbox']
            except KeyError:
                Lbox = self._Lbox
        else:
            halo_x = kwargs['halo_x']
            halo_y = kwargs['halo_y']
            halo_z = kwargs['halo_z']
            halo_hostid = kwargs['halo_hostid']
            halo_id = kwargs['halo_id']
            try:
                Lbox = kwargs['Lbox']
            except KeyError:
                Lbox = self._Lbox
        
        # get subhalo positions
        x = halo_x*1.0
        y = halo_y*1.0
        z = halo_z*1.0

        # get host halo positions 
        inds1, inds2 = crossmatch(halo_hostid, halo_id)
        # x-position
        halo_x[inds1] = halo_x[inds2]
        # y-position
        halo_y[inds1] = halo_y[inds2]
        # z-position
        halo_z[inds1] = halo_z[inds2]

        # get host halo orientation
        host_halo_axisA_x = halo_axisA_x
        host_halo_axisA_x[inds1] = halo_axisA_x[inds2]
        host_halo_axisA_y = halo_axisA_y
        host_halo_axisA_y[inds1] = halo_axisA_y[inds2]
        host_halo_axisA_z = halo_axisA_z
        host_halo_axisA_z[inds1] = halo_axisA_z[inds2]
        host_halo_mjor_axes = np.vstack((halo_axisA_x,halo_axisA_y,halo_axisA_z)).T

        # calculate radial positions
        vec_r, r = radial_distance(x, y, z, halo_x, halo_y, halo_z, Lbox)

        # rotate radial vectors arond halo major axis
        N = len(x)
        rot_angles = np.random.uniform(0.0, 2*np.pi, N)
        rot_axes = host_halo_mjor_axes
        rot_m = rotation_matrices_from_angles(rot_angles,rot_axes)

        new_vec_r = rotate_vector_collection(rot_m, vec_r)
        xx = new_vec_r[:,0]
        yy = new_vec_r[:,1]
        zz = new_vec_r[:,2]

        # move back into original cordinate system
        xx = halo_x + xx
        yy = halo_y + yy
        zz = halo_z + zz
        
        # account for PBCs
        mask = (xx < 0.0)
        xx[mask] = xx[mask] + Lbox[0]
        mask = (xx > Lbox[0])
        xx[mask] = xx[mask] - Lbox[0]
        mask = (yy < 0.0)
        yy[mask] = yy[mask] + Lbox[1]
        mask = (yy > Lbox[1])
        yy[mask] = yy[mask] - Lbox[1]
        mask = (zz < 0.0)
        zz[mask] = zz[mask] + Lbox[2]
        mask = (zz > Lbox[2])
        zz[mask] = zz[mask] - Lbox[2]

        if 'table' in kwargs.keys():
            # assign satellite galaxy positions
            try:
                mask = (table['gal_type']=='satellites')
            except KeyError:
                mask = np.array([True]*len(table))
                msg = ("`gal_type` not indicated in `table`.",
                       "The orientation is being assigned for all galaxies in the `table`.")
                print(msg)
            
            table['x'] = halo_x*1.0
            table['y'] = halo_y*1.0
            table['z'] = halo_z*1.0

            table['x'][mask] = xx[mask]
            table['y'][mask] = yy[mask]
            table['z'][mask] = zz[mask]

            table['halo_x'][mask] = halo_x[mask]
            table['halo_y'][mask] = halo_y[mask]
            table['halo_z'][mask] = halo_z[mask]

            return table
        else:
            x = xx
            y = yy
            z = zz
            return np.vstack((x,y,z)).T

    def assign_gal_type(self, **kwargs):
        """
        specify central and satellites
        """
        
        if 'table' in kwargs.keys():
            table = kwargs['table']
            upid = table['halo_upid']
        else:
            upid = kwargs['halo_upid']


        centrals = (upid == -1)
        satellites = (upid != -1)
        
        if 'table' in kwargs.keys():
            # assign galaxy type
            table['gal_type'] = 'satellites'
            table['gal_type'][centrals] = 'centrals'

            return table
        else:
            result = np.array(['satellites']*len(upid))
            result[centrals] = 'centrals'
            return result



def radial_distance(x, y, z, halo_x, halo_y, halo_z, Lbox):
        """
        calculate radial position of satellite galaxies
        """

        dx = (x - halo_x)
        mask = dx>Lbox[0]/2.0
        dx[mask] = dx[mask] - Lbox[0]
        mask = dx<-1.0*Lbox[0]/2.0
        dx[mask] = dx[mask] + Lbox[0]

        dy = (y - halo_y)
        mask = dy>Lbox[1]/2.0
        dy[mask] = dy[mask] - Lbox[1]
        mask = dy<-1.0*Lbox[1]/2.0
        dy[mask] = dy[mask] + Lbox[1]

        dz = (z - halo_z)
        mask = dz>Lbox[2]/2.0
        dz[mask] = dz[mask] - Lbox[2]
        mask = dz<-1.0*Lbox[2]/2.0
        dz[mask] = dz[mask] + Lbox[2]

        r = np.sqrt(dx**2+dy**2+dz**2)

        return np.vstack((dx, dy, dz)).T, r
