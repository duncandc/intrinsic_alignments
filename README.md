# Intrinsic Alignments

This is a project to measure and model the intrinsic alignments of galaxies in both observations and simulations.

Specifically, this is my fork of Duncan's [intrinsic_alignments](https://github.com/duncandc/intrinsic_alignments) package with fixes to some of the errors encountered while trying to run.


## Requirements

This project requires the following Python packages be installed:

* [numpy](http://www.numpy.org)
* [scipy](https://www.scipy.org)
* [astropy](http://www.astropy.org)
* The alignments_devel branch of Duncan's fork of [halotools](https://github.com/duncandc/halotools)
* My fork of [rotations](https://github.com/nvanalfen/rotations) currently only the test branch is updated
* [watson_dist](https://github.com/duncandc/watson_dist)


## Installation Instructions for Anaconda
I am currently working on simplifying the installation process for rotations.


Conda install is available for all except the specific version of halotools, rotations, and watson_dist. To install these from github using conda, do the following:
* conda install git
* conda install pip
* To install rotations you need a setup.py file, currently working on that
* To install the proper fork of halotools:
*   Make sure you have gcc version 5.3 or later. This is very important. The newest version will be the best option.
*   Activate your conda environment
*   Within your environment, run: pip install git+https://github.com/duncandc/halotools@alignments_devel
* To install watson_dist there are several options:
*   Run: pip install git+https://github.com/duncandc/watson_dist
*   OR clone the repo and in the directory with setup.py and run: conda develop .

Alignment correlation functions are in development in Duncan's personal halotools fork in a dedicated [branch](https://github.com/duncandc/halotools/tree/alignments_devel).  In addition to Duncan, this branch has contributions by:

* Patricia Larson
* Andrew Hearin
* Simon Samuroff


## Simulation Products


### Hydrodynics Simualtions

This propject uses the results of an independent repository, [Illustris_Shapes](https://github.com/duncandc/Illustris_Shapes), for simulated galaxy/halo shapes and orientation catalogs.

### Dark Matter Only Simulations

In addition to the Illustris DMO simulations, this project uses the dark matter only simualtions:

*  Bolshoi Planck

These halo catalogs for these simulations are available through Halotools, described [here](https://halotools.readthedocs.io/en/latest/quickstart_and_tutorials/quickstart_guides/working_with_halotools_provided_catalogs.html). 



