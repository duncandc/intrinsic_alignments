#!/usr/bin/env bash

# note that the dropbox links have been modified 
# so that each link trails with dl=1 instead of dl=0

# download shape catalogs
#wget -r -c -O ./shape_catalogs/download.gz https://www.dropbox.com/sh/ca2v1uv6lehdei2/AADros5uQ1WM8E0a-i0h7xv8a?dl=1
#tar -xvf ./shape_catalogs/download.gz -C ./shape_catalogs
#rm ./shape_catalogs/download.gz

# download value added galaxy catalogs
wget -r -c -O ./value_added_catalogs/download.gz https://www.dropbox.com/sh/7gtz6ejkz8egjv2/AACbdVC3CFmrdboxAzuwOyDda?dl=1
tar -xvf ./value_added_catalogs/download.gz -C ./value_added_catalogs
rm ./value_added_catalogs/download.gz

