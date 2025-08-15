# Load packages
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
import zarr 
import gsw  # (Gibbs Seawater Oceanography Toolkit) https://teos-10.github.io/GSW-Python/gsw.html

from numpy.linalg import lstsq
import seaborn as sns
from scipy.stats import gaussian_kde

from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
import pandas as pd

import dask 

import xrft

# Load the model
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

# Folder to store the figures
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin"

# Global font size setting for figures
plt.rcParams.update({'font.size': 16})

# Set spatial indices
face = 2
k_surf = 0
# i = slice(0,100,1) # Southern Ocean
# j = slice(0,101,1) # Southern Ocean
# i = slice(1000,1200,1) # Tropics
# j = slice(2800,3001,1) # Tropics

# i = slice(900,1300,1) # Tropics
# j = slice(2700,3101,1) # Tropics

# i=slice(450,760,1)
# j=slice(450,761,1)
# i=slice(671,864,1)   # icelandic_basin
# j=slice(2982,3419,1) # icelandic_basin

i=slice(527,1007,1)   # icelandic_basin -- larger domain
j=slice(2960,3441,1) # icelandic_basin -- larger domain

# Grid spacings in m
dxF = ds1.dxF.isel(face=face,i=i,j=j)
dyF = ds1.dyF.isel(face=face,i=i,j=j)

# Coordinate
lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)
depth = ds1.Z
depth_kp1 = ds1.Zp1

# Convert lat/lon from xarray to NumPy arrays
lat_vals = lat.values  # shape (j,)
lon_vals = lon.values  # shape (i,)

# Create 2D lat/lon meshgrid
lon2d, lat2d = np.meshgrid(lon_vals, lat_vals, indexing='xy')  # shape (j, i)

# Find the center location of selected region
lat_c = float(lat.mean().values)
lon_c = float(lon.mean().values)
# print(f"Center location: lat={lat_c}, lon={lon_c}")

# Find the time zone
tf = TimezoneFinder()
timezone_str = tf.timezone_at(lng=lon_c, lat=lat_c)
# print(f"Detected timezone: {timezone_str}")

# Convert UTC to Local time
time_utc = pd.to_datetime(ds1.time.values)
time_local = time_utc.tz_localize('UTC').tz_convert(timezone_str)
# print(time_local[:5])

# Set temporal indices:
indices_14 = [time_idx for time_idx, t in enumerate(time_local) if t.hour == 14]  # 2pm local time
indices_15 = [time_idx for time_idx, t in enumerate(time_local) if t.hour == 15]  # 3pm local time
indices_16 = [time_idx for time_idx, t in enumerate(time_local) if t.hour == 16]  # 4pm local time

time_inst = indices_15[0]   

nday_avg = 364                 # multiple-day average
time_avg = []
for time_idx in range(nday_avg):
    time_avg.extend([indices_14[time_idx], indices_15[time_idx], indices_16[time_idx]])
# time_avg = slice(0,24*nday_avg,1)  

start_hours = 132*24
# start_hours = 132*24 + 20*24
# start_hours = 1
time_avg = slice(start_hours,start_hours+24*nday_avg,1) 


############ Load data ############
# Read temperature and salinity data of the top 1000 m 
tt = ds1.Theta.isel(time=time_avg,face=face,i=i,j=j) # Potential temperature
ss = ds1.Salt.isel(time=time_avg,face=face,i=i,j=j)  # Practical salinity
ww = ds1.W.isel(time=time_avg,face=face,i=i,j=j)  # Practical salinity

# # Re-chunk time dimension for 12-hour average
tt = tt.chunk({'time': 12}) 
ss = ss.chunk({'time': 12})  
ww = ww.chunk({'time': 12})  
print(tt.chunks) 
# Build a lazy Dask graph — nothing is computed yet
tt_12h = tt.coarsen(time=12, boundary='trim').mean()
ss_12h = ss.coarsen(time=12, boundary='trim').mean()
ww_12h = ww.coarsen(time=12, boundary='trim').mean()
# Trigger computation
tt_12h = tt_12h.compute()
ss_12h = ss_12h.compute()
ww_12h = ww_12h.compute()

############ Compute Buoyancy ############
# Compute the potential density using GSW (Gibbs Seawater Oceanography Toolkit), with surface reference pressure 
SA_12h = gsw.conversions.SA_from_SP(ss_12h, depth, lon, lat) # Absolute salinity, shape (k, j, i)
CT_12h = gsw.conversions.CT_from_pt(SA_12h, tt_12h)              # Conservative temperature, shape (k, j, i)
p_ref = 0                                            # Reference pressure 
rho_12h = gsw.density.rho(SA_12h, CT_12h, p_ref)                 # Potential density, shape (k, j, i)

rho0 = 1000
gravity = 9.81
buoy_12h = -gravity*(rho_12h-rho0)/rho0

# interp W on to the vertical level of density
ww_12h_z = ww_12h.rename({'k_p1': 'Z'})
ww_12h_z = ww_12h_z.assign_coords(Z=depth_kp1.values)
ww_12h_interp = ww_12h_z.interp(Z=depth)

print(ww_12h_interp.dims)
print(buoy_12h.dims)

# Rechunk before spectral analysis
print(buoy_12h.chunks)
ww_12h_interp = ww_12h_interp.chunk({'i': -1, 'j': -1})
buoy_12h = buoy_12h.chunk({'i': -1, 'j': -1})
print(buoy_12h.chunks)
# Fill NaN data with 0
buoy_12h = buoy_12h.fillna(0)
ww_12h_interp = ww_12h_interp.fillna(0)
# cospectrum of w and b 
WB_cross_spectra = xrft.isotropic_cross_spectrum(ww_12h_interp, buoy_12h, dim=['i','j'], 
                                           detrend='linear', window='hann', truncate=True).compute().mean('time')


############ Make plots and save data ############
# freq_r = WB_cross_spectra.freq_r
# spec_vp = WB_cross_spectra * freq_r

dx = dxF.mean()
dy = dyF.mean()
dr = np.sqrt(dx**2/2 + dy**2/2)

k_r = WB_cross_spectra.freq_r/dr/1e-3
spec_vp = WB_cross_spectra * k_r


# Filter out large-scale (low wavenumber) components
k_r_filtered = k_r.where((k_r >= 2/500).compute(), drop=True)
spec_vp_filtered = spec_vp.where((k_r >= 2/500).compute(), drop=True)

from matplotlib.colors import TwoSlopeNorm

plt.figure(figsize=(8, 6))
vmax = np.abs(spec_vp_filtered.real).max().compute()
norm = TwoSlopeNorm(vcenter=0, vmin=-vmax, vmax=vmax)  

pc = plt.pcolormesh(k_r_filtered, WB_cross_spectra['Z'], spec_vp_filtered.real, shading='auto', cmap='RdBu',norm=norm)
plt.gca().invert_yaxis()  # Flip y-axis so depth increases downward
plt.xscale('log')
plt.xlabel(r'Wavenumber $k_r$ (cpkm)')
plt.ylabel('Depth (m)')
plt.title(r'$w,\ b$ Cross-Spectrum (VP)')
plt.colorbar(pc, label=r'Spectral density (m$^{2}$s$^{-3}$)')
plt.gca().invert_yaxis()
plt.grid(True, which='both', ls='--')
plt.tight_layout()
plt.savefig(f"{figdir}/wb_cross-spectrum_2D.png", dpi=150)
plt.close()


#### TO DO: COMPUTE MIXED LAYER AVERAGE
# k_r_max = WB_spectra_mld.freq_r.where(WB_spectra_mld*k_r == (WB_spectra_mld*k_r).max(),drop=True)/dx/1e-3
# L_max = 1/k_r_max




# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster(n_workers=64, threads_per_worker=1)
# client = Client(cluster)

# from dask import delayed, compute
# from numpy.linalg import lstsq

# # Rechunk to reduce per-task memory load
# ww_12h_interp = ww_12h_interp.chunk({'time': 1, 'i': -1, 'j': -1})
# buoy_12h = buoy_12h.chunk({'time': 1, 'i': -1, 'j': -1})

# # Cut time slices
# w_list = [ww_12h_interp.isel(time=t) for t in range(ww_12h_interp.sizes['time'])]
# b_list = [buoy_12h.isel(time=t) for t in range(buoy_12h.sizes['time'])]

# # Define function to run per time step
# def compute_cross_spec(w, b):
#     return xrft.isotropic_cross_spectrum(w, b, dim=['i', 'j'], detrend='linear', window='hann', truncate=True)

# # Submit each time step to the Dask scheduler
# futures = [client.submit(compute_cross_spec, w, b) for w, b in zip(w_list, b_list)]

# # Wait and gather results
# results_computed = client.gather(futures)

# # Combine and average spectra
# WB_cross_spectra = xr.concat(results_computed, dim='time').mean('time')





# # Step 3: use delayed with minimal graph
# @delayed
# def compute_one_time_spec_delayed(w, b):
#     return xrft.isotropic_cross_spectrum(w, b, dim=['i', 'j'], detrend='linear', window='hann', truncate=True)

# results_delayed = [compute_one_time_spec_delayed(w, b) for w, b in zip(w_list, b_list)]

# # Trigger parallel execution
# results_computed = compute(*results_delayed)

# # Combine to one DataArray and average
# WB_cross_spectra = xr.concat(results_computed, dim='time').mean('time')


# # Rechunk before spectral analysis
# buoy_12h = buoy_12h.chunk({'time': 1, 'i': -1, 'j': -1})
# ww_12h_interp = ww_12h_interp.chunk({'time': 1, 'i': -1, 'j': -1})

# # Compute cospectrum of w and b in parallel
# def compute_isotropic_spectrum(ds):
#     w = ds['w']
#     b = ds['b']
#     spec = xrft.isotropic_cross_spectrum(
#         w, b, dim=['i', 'j'],
#         detrend='linear',
#         window='hann',
#         truncate=True
#     )
#     return spec

# # combine two DataArray into one Dataset, to be used by map_blocks
# ds_pair = xr.Dataset({'w': ww_12h_interp, 'b': buoy_12h})

# # Compute the cross-spectra
# WB_spectra_all = xr.map_blocks(compute_isotropic_spectrum, ds_pair)

# # Time-average
# WB_cross_spectra = WB_spectra_all.mean('time').compute()


############ Compute vertical buoyancy flux ############

############ Compute w' and b' ############




############ Compute cross-spectra of w' and b' ############






# # # Re-chunk time dimension for long-time average
# tt = tt.chunk({'time': -1})  # Re-chunk to include all data points
# ss = ss.chunk({'time': -1})  # Re-chunk to include all data points
# ww = ww.chunk({'time': -1})  # Re-chunk to include all data points
# print(tt.chunks) 
# # Build a lazy Dask graph — nothing is computed yet
# tt_mean = tt.mean(dim='time')
# ss_mean = ss.mean(dim='time')
# ww_mean = ww.mean(dim='time')
# # Trigger computation
# tt_mean = tt_mean.compute()
# ss_mean = ss_mean.compute()
# ww_mean = ww_mean.compute()
