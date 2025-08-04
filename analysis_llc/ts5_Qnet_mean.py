##### Time series of the following variables:
#####
##### Qnet (net surface heat flux into the ocean),
##### Hml (mixed-layer depth), 
##### TuH (horizontal Turner angle), 
##### TuV (vertical Turner angle),
##### wb_cros (variance-perserving cross-spectrum of vertical velocity and buoyancy)
##### wbmin (the minimum of wb_cros)
##### Lmax (the horizontal length scale corresponds to wbmin), 
##### Dmax (the depth corresponds to wbmin), 
##### gradSSH (absolute gradient of sea surface height anomaly), etc.
#####
##### Step 1: compute 12-hour averages of temperature, salinity, and vertical velocity, save as .nc files
##### Step 2: compute 7-day averages of potential density, alpha, beta, Hml, save as .nc files
##### Step 3: compute wb_cros using the 12-hour averages, and then compute the 7-day averaged wb_cros 
##### Step 4: plot wb_cros of each week, compute wbmin, Lmax, Dmax
##### Step 5: compute 7-day movmean of Qnet
##### Step 6: compute TuH and TuV

# Load packages
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
import zarr
import gsw  # Gibbs Seawater Toolkit: https://teos-10.github.io/GSW-Python/gsw.html

from numpy.linalg import lstsq
import seaborn as sns
from scipy.stats import gaussian_kde

from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
import pandas as pd

import dask
from dask.distributed import Client, LocalCluster

# Dask cluster setup
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print(client.dashboard_link)

# Load model data (only needed variables)
ds1 = xr.open_zarr(
    '/orcd/data/abodner/003/LLC4320/LLC4320',
    consolidated=False,
    chunks={}  # use on-demand chunking
)[['oceQnet', 'dxF', 'dyF', 'XC', 'YC', 'time', 'Z']]

# Output folder
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin"
output_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin"

# Global font settings
plt.rcParams.update({'font.size': 16})

# Grid selections
face = 2
k_surf = 0
i = slice(527, 1007, 60)
j = slice(2960, 3441, 60)

# Coordinates
lat = ds1.YC.isel(face=face, i=1, j=j)
lon = ds1.XC.isel(face=face, i=i, j=1)
depth = ds1.Z

lat_c = float(lat.mean().values)
lon_c = float(lon.mean().values)

# Local time zone detection
tf = TimezoneFinder()
timezone_str = tf.timezone_at(lng=lon_c, lat=lat_c)

# Time selection 
# nday_avg = 364
nday_avg = 7
start_hours = 49 * 24
end_hours = start_hours + 24 * nday_avg
time_avg = slice(start_hours, end_hours, 1)

# Load and chunk Qnet (use larger time chunks for averaging)
oceQnet = ds1.oceQnet.isel(time=time_avg, face=face, i=i, j=j).chunk({'time': 24, 'i': -1, 'j': -1})

# Area-mean time series (still lazy)
oceQnet_mean = oceQnet.mean(dim=['i', 'j']).persist()

# Assign datetime coordinates for time
oceQnet_mean['time'] = pd.to_datetime(ds1.time.isel(time=time_avg).values)

# Resample to daily mean and apply 7-day smoothing
oceQnet_daily = oceQnet_mean.resample(time='1D').mean()
oceQnet_smooth = oceQnet_daily.rolling(time=7, center=True).mean()

# Compute both in parallel
oceQnet_daily_vals, oceQnet_smooth_vals = dask.compute(oceQnet_daily, oceQnet_smooth)

# Save to NetCDF with time as coordinate
ds_qnet_daily = xr.Dataset(
    {
        "qnet_daily_avg": ("time", oceQnet_daily_vals.data),
        "qnet_7day_smooth": ("time", oceQnet_smooth_vals.data),
    },
    coords={
        "time": oceQnet_daily_vals.time.data,
    },
    attrs={
        "description": "Daily averaged and 7-day smoothed net surface heat flux (Qnet)",
        "source": "Processed from LLC4320 model data",
    }
)

output_nc_path = f"{output_dir}/qnet_daily_7day_smooth.nc"
ds_qnet_daily.to_netcdf(output_nc_path)
print(f"Saved NetCDF to: {output_nc_path}")

# Plot daily and smoothed time series
plt.figure(figsize=(10, 4))
plt.plot(ds_qnet_daily.time, ds_qnet_daily.qnet_daily_avg, label='Daily Avg Qnet', color='tab:red', alpha=0.4)
plt.plot(ds_qnet_daily.time, ds_qnet_daily.qnet_7day_smooth, label='7-day Smoothed Qnet', color='tab:blue')

plt.axhline(0, color='k', linestyle='--', linewidth=1)
plt.xlabel('Date')
plt.ylabel('Qnet [W/m²]')
plt.title('Net Surface Heat Flux into the Ocean')
plt.grid(True)
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{figdir}/oceQnet_daily_7day_smooth.png", dpi=150)
plt.close()