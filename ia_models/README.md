# Models for Intrinsic Alignmentmets

This directory contains classes and functions for modelling the intrinsic alignment(s) of galaxies.  Broadly, the code here can be split into tools for building empircal models for IA and my own implementations of power spectrum and fitting function based methods for prediciting various alignment correlation functions.

## Empirical Models
The code necessary to build empirical models for IA are located in the following files:
* ia\_model\_componenets.py
* anisotropic\_nfw\_phase\_space.py
* occupation\_models.py
* watson\_distribution.py

## Linear Alingnent Models
The code necessary to build linear and non-linear alignment models are located in the following files:
* linear\_alignment\_models.py
* hrh\_model.py

## Utilities
These utility functions are used by various components.
* utils.py
* cosmo\_utils.py

## Tests
Some basic tests are preformed by code located in the tests directory.  These can be run using the py.test package.

## Examples
Please see the tutorial notebooks in the notebooks directory for examples of how various components are applied.

## Required External Packages
This code requires a number of additional packages.  Some components can be used without installing all the packages.  These packages are listed below:

* [halotools](https://halotools.readthedocs.io/en/latest/)
* [astropy](http://www.astropy.org)
* [pycamb](http://camb.readthedocs.io/en/latest/)
* [pyfftlog](https://github.com/McWilliamsCenter/pyfftlog)
* [hmf](http://hmf.readthedocs.io/en/latest/)
