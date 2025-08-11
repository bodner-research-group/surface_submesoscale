import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import zarr
import gsw  # Gibbs Seawater Toolkit: https://teos-10.github.io/GSW-Python/gsw.html
import dask 

from timezonefinder import TimezoneFinder
import pandas as pd

from dask.distributed import Client, LocalCluster
from dask import compute

# ========== Dask cluster setup ==========
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)


# Load model data
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

# Output folder
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin"
output_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin"

# Plot font
plt.rcParams.update({'font.size': 16})

# Grid selections
face = 2
i = slice(527, 1007, 1)
j = slice(2960, 3441, 1)

# Coordinates
lat = ds1.YC.isel(face=face, i=1, j=j)
lon = ds1.XC.isel(face=face, i=i, j=1)
lat_c = float(lat.mean().values)
lon_c = float(lon.mean().values)

# Local timezone (not used below, but kept for possible datetime localization)
tf = TimezoneFinder()
timezone_str = tf.timezone_at(lng=lon_c, lat=lat_c)

# Time selection
nday_avg = 364
start_hours = 49 * 24
end_hours = start_hours + 24 * nday_avg
time_avg = slice(start_hours, end_hours,1)

# Extract variables and compute daily + 7-day means (no Dask)
oceQnet = ds1.oceQnet.isel(time=time_avg, face=face, i=i, j=j).chunk({'time': 24, 'j': -1, 'i': -1}) 
oceFWflx = ds1.oceFWflx.isel(time=time_avg, face=face, i=i, j=j).chunk({'time': 24, 'j': -1, 'i': -1}) 
print(oceQnet.chunks)

oceQnet = oceQnet.load()
oceFWflx = oceFWflx.load()

# Area-mean over i, j
oceQnet_mean = oceQnet.mean(dim=["i", "j"])
oceFWflx_mean = oceFWflx.mean(dim=["i", "j"])

# Assign datetime coordinates
time_vals = pd.to_datetime(ds1.time.isel(time=time_avg).values)
oceQnet_mean["time"] = time_vals
oceFWflx_mean["time"] = time_vals

# Coarsen to daily averages (24 hourly steps)
oceQnet_daily_vals = oceQnet_mean.coarsen(time=24, boundary='trim').mean()
oceFWflx_daily_vals = oceFWflx_mean.coarsen(time=24, boundary='trim').mean()

# Apply 7-day moving average
oceQnet_smooth_vals = oceQnet_daily_vals.rolling(time=7, center=True).mean()
oceFWflx_smooth_vals = oceFWflx_daily_vals.rolling(time=7, center=True).mean()

# Compute
oceQnet_daily_vals, oceFWflx_daily_vals, oceQnet_smooth_vals, oceFWflx_smooth_vals = compute(
    oceQnet_daily_vals, oceFWflx_daily_vals, oceQnet_smooth_vals, oceFWflx_smooth_vals
)

# Merge into one dataset
ds_combined = xr.Dataset(
    {
        "qnet_daily_avg": oceQnet_daily_vals,
        "qnet_7day_smooth": oceQnet_smooth_vals,
        "fwflx_daily_avg": oceFWflx_daily_vals,
        "fwflx_7day_smooth": oceFWflx_smooth_vals,
    },
    attrs={
        "description": "Daily and 7-day smoothed time series of surface fluxes",
        "source": "Processed from LLC4320 model data",
    }
)

# Save to NetCDF
output_nc_path = f"{output_dir}/qnet_fwflx_daily_7day.nc"
ds_combined.to_netcdf(output_nc_path)
print(f"Saved NetCDF to: {output_nc_path}")







import xarray as xr
import matplotlib.pyplot as plt
import gsw

# Load dataset
ds_combined = xr.open_dataset("/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/qnet_fwflx_daily_7day.nc")

# Figure output path
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/icelandic_basin"

# Global plot style
plt.rcParams.update({'font.size': 16})

# Constants for buoyancy flux calculation
gravity = 9.81
SA = 35.2
CT = 6
p = 0
rho0 = 999.8  # kg/m^3
Cp = 3975     # J/kg/K

beta = gsw.beta(SA, CT, p)
alpha = gsw.alpha(SA, CT, p)

# Compute buoyancy flux
Bflux_daily_avg = gravity * alpha * ds_combined.qnet_daily_avg / (rho0 * Cp) + gravity * beta * ds_combined.fwflx_daily_avg * SA / rho0
Bflux_7day_smooth = gravity * alpha * ds_combined.qnet_7day_smooth / (rho0 * Cp) + gravity * beta * ds_combined.fwflx_7day_smooth * SA / rho0

# Create combined figure with 3 panels
fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

# Panel (a): Qnet
axs[0].plot(ds_combined.time, ds_combined.qnet_daily_avg, label='Daily Avg Qnet', color='tab:red', alpha=0.4)
axs[0].plot(ds_combined.time, ds_combined.qnet_7day_smooth, label='7-day Smoothed Qnet', color='tab:blue')
axs[0].axhline(0, color='k', linestyle='--', linewidth=1)
axs[0].set_ylabel('Qnet [W/m²]')
axs[0].set_title('(a) Net Surface Heat Flux into the Ocean')
axs[0].grid(True)
axs[0].legend()

# Panel (b): FWflx
axs[1].plot(ds_combined.time, ds_combined.fwflx_daily_avg, label='Daily Avg FWflx', color='tab:red', alpha=0.4)
axs[1].plot(ds_combined.time, ds_combined.fwflx_7day_smooth, label='7-day Smoothed FWflx', color='tab:blue')
axs[1].axhline(0, color='k', linestyle='--', linewidth=1)
axs[1].set_ylabel('FWflx [kg/m²/s]')
axs[1].set_title('(b) Net Surface Fresh Water Flux')
axs[1].grid(True)
axs[1].legend()

# Panel (c): Bflux
axs[2].plot(ds_combined.time, Bflux_daily_avg, label='Daily Avg Bflx', color='tab:red', alpha=0.4)
axs[2].plot(ds_combined.time, Bflux_7day_smooth, label='7-day Smoothed Bflx', color='tab:blue')
axs[2].axhline(0, color='k', linestyle='--', linewidth=1)
axs[2].set_ylabel('Bflx [m²/s³]')
axs[2].set_title('(c) Net Surface Buoyancy Flux')
axs[2].grid(True)
axs[2].legend()

# Final adjustments
# plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f"{figdir}/combined_surface_fluxes.png", dpi=150)
plt.close()
