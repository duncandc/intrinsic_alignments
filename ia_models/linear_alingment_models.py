from hmf import hmf_cosmo, growth_factor
from camb import model, initialpower
import numpy as np
from astropy import units as u

# get growth factor function
gf = growth_factor.GrowthFactor(c)

# set up a new set of parameters for CAMB
pars = camb.CAMBparams()

def mean_density(cosmo, z):
    """
    mean density of the universe
    
    Paramaters
    ----------
    z : array_like
        redshift
    
    cosmo : astropy.cosmology object
    
    Returns
    -------
    rho_b : numpy.array
         mean density of the universe at redshift z in Msol/Mpc^3
    """
    
    scale_factor = 1.0/(1.0+z)
    
    rho = (3.0/(8.0*np.pi*const.G))*(cosmo.H(z)**2)*(cosmo.Om(z)*scale_factor**(-3))
    rho = rho.to(u.M_sun  / u.parsec**3)*((10**6)**3)
    
    return rho


def linear_power_spectrum(cosmo, z, lmax=2500, minkh=1e-4, maxkh=1):
    """
    Parameters
    ==========

    Returns
    =======
    k : numpy.array
        k for which the power specturm is calculated

    z : numpy.array
        redshifts for which the power epctrum is calculated

    pk : numpy.array
        power spectrum(s) at the specified k for each redshift
    """

    z = np.atleast_1D(z)

    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122, mnu=0.06, omk=0, tau=0.06)
    pars.InitPower.set_params(ns=0.965, r=0)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)

    # calculate results for these parameters
    results = camb.get_results(pars)

    # now get matter power spectra and sigma8 at the specified  redshift
    pars = camb.CAMBparams()
    pars.set_cosmology(H0=67.5, ombh2=0.022, omch2=0.122)
    pars.set_dark_energy() #re-set defaults
    pars.InitPower.set_params(ns=0.965)
    # not non-linear corrections couples to smaller scales than you want
    pars.set_matter_power(redshifts=z, kmax=maxkh*2)

    # linear spectra
    pars.NonLinear = model.NonLinear_none
    results = camb.get_results(pars)
    kh, z, pk = results.get_matter_power_spectrum(minkh=minkh, maxkh=maxkh, npoints=200)

    return kh, z, pk


def linear_growth_factor(cosmo, z):
    """
    linear growth factor normalized to 1.0 at z=0.0
    """
    
    gf_f = growth_factor.GrowthFactor(hmf_cosmo)
    
    return gf_f(z)


def P_II_factor(cosmo, z):
    """
    alignment shape-shape linear power spectrum factor
    """
    C_1 = 5*10**(-14)
    return (C_1*mean_density(cosmo, z)/((1.0+z)*linear_growth_factor(cosmo, z)))**2



