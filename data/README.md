# Data

This directory stores data needed for and produced by this project.


## Downloading Data

Some of the data products needed for this project can be downloaded by running a set of scripts.

The catalogs produced in the [Illustris_Shapes](https://github.com/duncandc/Illustris_Shapes) project are available on Dropbox.  The catalogs can be downloaded using the `./Illustris/download_data.sh` script.

Halotools provides some processed DMO halo catalogs, which this project uses.  For example, to download the Bolshoi Planck z=0.0 catalog, execute the following command in the Halotools base directory:

```
$user: python scripts/download_additional_halocat.py bolplanck rockstar halotools_v0p4 0.0
```

This will cache the halo catalogs.  See the Halotools documentation is you wish to modify the location the cahced catalogs are saved.

