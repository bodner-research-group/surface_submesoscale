# Calculate the horizontal Turner Angle following Johnson et al. (2016) JPO
# Using instantaneous output or time-averages

# Load packages
import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr 
import dask 
import gsw  # (Gibbs Seawater Oceanography Toolkit) https://teos-10.github.io/GSW-Python/gsw.html

# Load the model
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

# Folder to store the figures
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/face01_test1"

# Global font size setting for figures
plt.rcParams.update({'font.size': 16})

# Set indices
face = 1
nday_avg = 7                 # 7-day average
time_avg = slice(0,24*nday_avg,1)  
time_inst = 0
k_surf = 0
i = slice(0,100,1) # Southern Ocean
j = slice(0,101,1) # Southern Ocean
# i = slice(1000,1200,1) # Tropics
# j = slice(2800,3001,1) # Tropics
# Use different sizes for Nx and Ny to avoid using wrong dimensions

# Coordinate
lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)
depth = ds1.Z

# Convert lat/lon from xarray to NumPy arrays
lat_vals = lat.values  # shape (j,)
lon_vals = lon.values  # shape (i,)

# Create 2D lat/lon meshgrid
lon2d, lat2d = np.meshgrid(lon_vals, lat_vals, indexing='xy')  # shape (j, i)


##############################################################
### 0. Load instantaneous output or compute time averages  ###
##############################################################

################ If use instantaneous output
tt = ds1.Theta.isel(time=time_inst,face=face,i=i,j=j) # Potential temperature
ss = ds1.Salt.isel(time=time_inst,face=face,i=i,j=j)  # Practical salinity
################

# ################ If calculate time averages
# # Read temperature and salinity data of the top 1000 m 
# tt = ds1.Theta.isel(time=time_avg,face=face,i=i,j=j) # Potential temperature
# ss = ds1.Salt.isel(time=time_avg,face=face,i=i,j=j)  # Practical salinity
# # eta = ds1.Eta.isel(time=time,face=face,i=i,j=j)  # Surface Height Anomaly
# print(tt.chunks) 

# # Re-chunk time dimension to 7-day blocks for efficient averaging
# tt = tt.chunk({'time': 24*nday_avg}) 
# ss = ss.chunk({'time': 24*nday_avg})
# print(tt.chunks) 

# # Compute time averages
# # Build a lazy Dask graph — nothing is computed yet
# tt_mean = tt.mean(dim='time')
# ss_mean = ss.mean(dim='time')

# # Trigger computation
# tt = tt_mean.compute()
# ss = ss_mean.compute()
# ################
