# Calculate the horizontal Turner Angle following Johnson et al. (2016) JPO
# Using instantaneous output or time-averages

# 1. Use plain fit to compute the gradients
# 2. Calculate the horizontal Turner Angle
# 3. Calculate the Kernel PDF distribution 


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
# from dask.distributed import Client, LocalCluster
# cluster = LocalCluster(n_workers=64, threads_per_worker=1)
# client = Client(cluster)

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
# i=slice(450,760,1)
# j=slice(450,761,1)
i=slice(671,864,2)   # icelandic_basin
j=slice(2982,3419,2) # icelandic_basin

# Grid spacings in m
dxF = ds1.dxF.isel(face=face,i=i,j=j)
dyF = ds1.dyF.isel(face=face,i=i,j=j)

# Coordinate
lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)
depth = ds1.Z

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
 

nday_avg = 200                 # multiple-day average
start_hours = 100*24
time_avg = slice(start_hours,start_hours+24*nday_avg,1) 

######### Load data #########
oceQnet = ds1.oceQnet.isel(time=time_avg,face=face,i=i,j=j)   # net surface heat flux into the ocean (+=down), >0 increases theta

oceQnet = oceQnet.chunk({'time': -1})  # Re-chunk to include all data points
print(oceQnet.chunks) 

# Compute spatial averages
# Build a lazy Dask graph — nothing is computed yet
oceQnet_mean = oceQnet.mean(dim=['i', 'j'])

# Trigger computation
oceQnet_timeseries = oceQnet_mean.compute()

# Get time values
time_local_avg = ds1.time.isel(time=time_avg).values

# plot
plt.figure(figsize=(10, 4))
plt.plot(time_local_avg, oceQnet_timeseries, label='Net surface heat flux (Qnet)', color='tab:red')

plt.axhline(0, color='k', linestyle='--', linewidth=1)

plt.xlabel('Date')
plt.ylabel('Qnet [W/m²]')
plt.title('Surface Heat Flux Time Series (Area-averaged)')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig(f"{figdir}/oceQnet_timeseries.png", dpi=150)
plt.close()

# Convert time index to datetime (localized) and group by date
time_local_avg_full = pd.to_datetime(time_local_avg).tz_localize('UTC').tz_convert(timezone_str)
df_qnet = pd.DataFrame({'time': time_local_avg_full, 'qnet': oceQnet_timeseries.values})
df_qnet['date'] = df_qnet['time'].dt.date

# Compute daily mean
df_daily = df_qnet.groupby('date').mean()

# Apply 7-day rolling mean
df_daily['qnet_smooth'] = df_daily['qnet'].rolling(window=7, center=True).mean()

# Plot daily averaged and smoothed Qnet
plt.figure(figsize=(10, 4))
plt.plot(df_daily.index, df_daily['qnet'], label='Daily Avg Qnet', color='tab:red', alpha=0.4)
plt.plot(df_daily.index, df_daily['qnet_smooth'], label='7-day Smoothed Qnet', color='tab:blue')

plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Qnet [W/m²]')
plt.title('Net Surface Heat Flux into the Ocean')
plt.grid(True)
plt.legend()
# plt.tight_layout()
plt.xticks(rotation=45)
plt.savefig(f"{figdir}/oceQnet_daily_7day_smooth.png", dpi=150)
plt.close()