"""
utility functions for MBII value added subfind catalogs
"""

from __future__ import print_function, division
from astropy.table import Table
from halotools.utils import crossmatch
import numpy as np
import os
import re
import h5py
from astropy.cosmology import FlatLambdaCDM

DATA_DIRECTORY = os.path.dirname(os.path.realpath(__file__))+'/'
BASE_FNAME = 'MB2_subfind_z_'

def luminosity_to_magnitude(arr):
    """
    convert band luminosities to magnitudes
    """
    bandlum = arr*(10.0**28.0)
    bandflux = bandlum/(4.0*(np.pi)*(1.0*10**38)*(3.08567758**2))

    with np.errstate(divide='ignore'):
        return np.where(bandflux>0.0, -2.5*(np.log10(bandflux))-48.6, np.zeros_like(bandflux))

def central_satellite(subfind_catalog):
    """
    function specify central/satellite flags
    """

    # most massive subfind halo
    cen_mask = (subfind_catalog['halos.central'] == 1)
    sat_mask = (subfind_catalog['halos.central'] == 0)

    return cen_mask, sat_mask

def delta_vir(z, cosmo, wrt='background'):
    """
    The average over-density of a collapsed dark matter halo. 
    fitting function from Bryan & Norman (1998)
    
    Paramaters
    ----------
    z : array_like
        redshift
    
    cosmo : astropy.cosmology object
    
    Returns
    -------
    delta_vir : numpy.array
        average density with respect to the mean density of the Universe
    """
    
    z = np.atleast_1d(z)
    
    x = cosmo.Om(z)-1.0
    
    if wrt=='critical':
        return (18.0*np.pi**2 + 82.0*x - 39.0*x**2)
    elif wrt=='background':
        return (18.0*np.pi**2 + 82.0*x - 39.0*x**2)/cosmo.Om(z)

def make_catalog(filepath, redshift, cosmo):
    """
    process database formatted subfind catalog 
    """
    
    try:
        subfind_catalog = Table.read(filepath, path='hydro_full')
    except IOError:
        subfind_catalog = Table.read(filepath, path='catalog')

    catalog = Table()

    # unique ID
    catalog['id'] = subfind_catalog['halos.subfindId'].astype('int')

    # define an ID common to all galaxies in the same host halo
    catalog['host_halo_id'] = subfind_catalog['groups.groupId'].astype('int')
    
    # position and velocity of galaxies
    catalog['x'] = subfind_catalog['halos.x']/1000.0
    catalog['y'] = subfind_catalog['halos.y']/1000.0
    catalog['z'] = subfind_catalog['halos.z']/1000.0
    catalog['vx'] = subfind_catalog['halos.vx']
    catalog['vy'] = subfind_catalog['halos.vy']
    catalog['vz'] = subfind_catalog['halos.vz']
    
    # baryonic mass components
    catalog['m_star'] = subfind_catalog['halos.m_star']* 10.0**10.0
    catalog['m_dm']   = subfind_catalog['halos.m_dm']  * 10.0**10.0
    catalog['m_bh']   = subfind_catalog['halos.m_bh']  * 10.0**10.0
    catalog['m_gas']  = subfind_catalog['halos.m_gas'] * 10.0**10.0

    # magnitudes
    catalog['SDSS_g'] = luminosity_to_magnitude(subfind_catalog['prop.SDSS_g'])
    catalog['SDSS_i'] = luminosity_to_magnitude(subfind_catalog['prop.SDSS_i'])
    catalog['SDSS_r'] = luminosity_to_magnitude(subfind_catalog['prop.SDSS_r'])
    catalog['SDSS_u'] = luminosity_to_magnitude(subfind_catalog['prop.SDSS_u'])
    catalog['SDSS_z'] = luminosity_to_magnitude(subfind_catalog['prop.SDSS_z'])

    # other galaxy properties
    catalog['ssfr'] = np.divide(subfind_catalog['prop.sfr'],catalog['m_star'],
                                out=np.zeros_like(subfind_catalog['prop.sfr']),
                                where=catalog['m_star']!=0.0)
    catalog['b_to_t'] =  subfind_catalog['prop.btr']

    # identify central and satellite galaxies
    cen, sat = central_satellite(subfind_catalog)
    catalog['central'] = 0
    catalog['central'][cen] = 1
    catalog['satellite'] = 0
    catalog['satellite'][sat] = 1

    # dark matter mass of host (sub-)haloes
    catalog['halo_mass'] = 0.0
    catalog['halo_mass'][cen] = subfind_catalog['groups.m_dm'][cen] * 10.0**10.0
    catalog['halo_mass'][sat] = subfind_catalog['halos.m_dm'][sat] * 10.0**10.0

    # host halo properties
    inds1, inds2 = crossmatch(catalog['host_halo_id'], catalog['host_halo_id'][cen])
    catalog['host_halo_mass'] = 0.0
    catalog['host_halo_mass'][inds1] = catalog['halo_mass'][cen][inds2]

    # virial radius
    delta = delta_vir(redshift, cosmo, wrt='critical')
    rho_bar = delta*cosmo.critical_density(redshift).to('Msun / Mpc^3')
    catalog['host_virial_radius'] = (catalog['host_halo_mass']/((4.0/3.0)*np.pi*rho_bar))**(1.0/3.0)

    # galaxy shapes
    catalog['shapesStar.q3d'] = subfind_catalog['shapesStar.q3d']
    catalog['shapesStar.q2d'] = subfind_catalog['shapesStar.q2d']
    catalog['shapesStar.s3d'] = subfind_catalog['shapesStar.s3d']
    
    catalog['shapesStar.a3d_x'] = subfind_catalog['shapesStar.a3d_x']
    catalog['shapesStar.a3d_y'] = subfind_catalog['shapesStar.a3d_y']
    catalog['shapesStar.a3d_z'] = subfind_catalog['shapesStar.a3d_z']

    catalog['shapesStar.b3d_x'] = subfind_catalog['shapesStar.b3d_x']
    catalog['shapesStar.b3d_y'] = subfind_catalog['shapesStar.b3d_y']
    catalog['shapesStar.b3d_z'] = subfind_catalog['shapesStar.b3d_z']

    catalog['shapesStar.c3d_x'] = subfind_catalog['shapesStar.c3d_x']
    catalog['shapesStar.c3d_y'] = subfind_catalog['shapesStar.c3d_y']
    catalog['shapesStar.c3d_z'] = subfind_catalog['shapesStar.c3d_z']

    catalog['shapesStar.a2d_x'] = subfind_catalog['shapesStar.a2d_x']
    catalog['shapesStar.a2d_y'] = subfind_catalog['shapesStar.a2d_y']

    catalog['shapesStar.b2d_x'] = subfind_catalog['shapesStar.b2d_x']
    catalog['shapesStar.b2d_y'] = subfind_catalog['shapesStar.b2d_y']

    # halo shapes
    catalog['shapesDM.q3d'] = subfind_catalog['shapesDM.q3d']
    catalog['shapesDM.q2d'] = subfind_catalog['shapesDM.q2d']
    catalog['shapesDM.s3d'] = subfind_catalog['shapesDM.s3d']

    catalog['shapesDM.a3d_x'] = subfind_catalog['shapesDM.a3d_x']
    catalog['shapesDM.a3d_y'] = subfind_catalog['shapesDM.a3d_y']
    catalog['shapesDM.a3d_z'] = subfind_catalog['shapesDM.a3d_z']

    catalog['shapesDM.b3d_x'] = subfind_catalog['shapesDM.b3d_x']
    catalog['shapesDM.b3d_y'] = subfind_catalog['shapesDM.b3d_y']
    catalog['shapesDM.b3d_z'] = subfind_catalog['shapesDM.b3d_z']

    catalog['shapesDM.c3d_x'] = subfind_catalog['shapesDM.c3d_x']
    catalog['shapesDM.c3d_y'] = subfind_catalog['shapesDM.c3d_y']
    catalog['shapesDM.c3d_z'] = subfind_catalog['shapesDM.c3d_z']

    catalog['shapesDM.a2d_x'] = subfind_catalog['shapesDM.a2d_x']
    catalog['shapesDM.a2d_y'] = subfind_catalog['shapesDM.a2d_y']

    catalog['shapesDM.b2d_x'] = subfind_catalog['shapesDM.b2d_x']
    catalog['shapesDM.b2d_y'] = subfind_catalog['shapesDM.b2d_y']

    # make cut on catalog
    mask = (catalog['m_star'] > 6.5*10**7.0)
    catslog = catalog[mask]

    # identify orphans
    mask = (catalog['halo_mass'] == 0.0)
    catalog['orphan'] = 0
    catalog['orphan'][mask] = 1

    return catalog

class MBII_galaxy_catalog(object):
    """
    MBII galaxy catalog object
    """
    def __init__(self, redshift, dz_tol=0.05, **kwargs):
        """
        """

        filename, closest_redshift = match_MBII_subfind_filename(redshift)
        if np.fabs(closest_redshift-redshift)>dz_tol:
            msg = ('no catalog found within redshit tolerance. closest redshift is {0}'.format(closest_redshift))
            raise ValueError(msg)
        
        self.cosmo =  FlatLambdaCDM(H0=70.1, Om0=0.275, Ob0=0.046)
        self.redshift = closest_redshift
        self.Lbox = 100.0
        self.dm_particle_mass = 1.1*10**7.0

        # read in halo catalog
        filepath = DATA_DIRECTORY + filename
        self.galaxy_table = make_catalog(filepath, self.redshift, self.cosmo)


def match_MBII_subfind_filename(redshift):
    """
    find closest subfind catalog

    Returns
    =======
    fname, redshift
    """
    files = os.listdir(DATA_DIRECTORY)
    possible_files = [x for x in files if BASE_FNAME in x]
    tails = [str.split('_')[-1] for str in possible_files]
    str_redshifts = [str.split('.')[0] +'.'+ str.split('.')[1]  for str in tails]
    flt_redshifts = np.array(str_redshifts).astype('float')

    ind = np.argmin(np.fabs(flt_redshifts - redshift))
    
    return possible_files[ind], flt_redshifts[ind]




