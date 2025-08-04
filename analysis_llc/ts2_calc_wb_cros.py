##### Time series of the following variables:
#####
##### Qnet (net surface heat flux into the ocean),
##### Hml (mixed-layer depth), 
##### TuH (horizontal Turner angle), 
##### TuV (vertical Turner angle),
##### wb_cros (variance-perserving cross-spectrum of vertical velocity and buoyancy), 
##### Lmax (the horizontal length scale corresponds to wb_cros minimum), 
##### Dmax (the depth corresponds to wb_cros minimum), 
##### gradSSH (absolute gradient of sea surface height anomaly), etc.
#####
##### Step 1: compute 12-hour averages of temperature, salinity, and vertical velocity, save as .nc files
##### Step 2: compute 7-day averages of potential density, alpha, beta, Hml, save as .nc files
##### Step 3: compute wb_cros using the 12-hour averages, and then compute the 7-day averaged wb_cros using a sliding window


# ========== Imports ==========
import xarray as xr
import numpy as np
import gsw
import os
from glob import glob
from dask.distributed import Client, LocalCluster

# ========== Dask cluster setup ==========
cluster = LocalCluster(
    n_workers=64,             # 1 worker per CPU core
    threads_per_worker=1,     # avoid Python GIL
    memory_limit="5GB"        # total = 320 GB < 386 GB limit
)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Open LLC4320 Dataset ==========
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

face = 2
i = slice(527, 1007)
j = slice(2960, 3441)
lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)
print(lat.chunks)
print(lon.chunks)

lon = lon.chunk({'i': -1})  # Re-chunk to include all data points
print(lon.chunks)

depth = ds1.Z

# ========== Broadcast lon, lat, depth ==========
lon_b, lat_b = xr.broadcast(lon, lat)
depth3d, _, _ = xr.broadcast(depth, lon_b, lat_b)

