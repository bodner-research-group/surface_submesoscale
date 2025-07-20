# Load packages
import multiprocessing as mp
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
import zarr 
import dask 
import gsw  # (Gibbs Seawater Oceanography Toolkit) https://teos-10.github.io/GSW-Python/gsw.html

from numpy.linalg import lstsq
import seaborn as sns
from scipy.stats import gaussian_kde

from timezonefinder import TimezoneFinder
from datetime import datetime
import pytz
import pandas as pd



# Load the model
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

# Folder to store the figures
figdir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/face01_day1_3pm"

# Global font size setting for figures
plt.rcParams.update({'font.size': 16})

# Set spatial indices
face = 1
k_surf = 0
i = slice(0,100,1) # Southern Ocean
j = slice(0,101,1) # Southern Ocean
# i = slice(1000,1200,1) # Tropics
# j = slice(2800,3001,1) # Tropics
i_g = i
j_g = j

# Grid spacings in m
dxF = ds1.dxF.isel(face=face,i=i,j=j)
dyF = ds1.dyF.isel(face=face,i=i,j=j)

# Coordinate
lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)

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
indices_15 = [i for i, t in enumerate(time_local) if t.hour == 15]  # 3pm local time
# print(indices_15)
# print(len(indices_15))

time_inst = indices_15[0]                



# Grid spacings in m
dxF = ds1.dxF.isel(face=face,i=i,j=j)
dyF = ds1.dyF.isel(face=face,i=i,j=j)

# Coordinate
lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)

# Convert lat/lon from xarray to NumPy arrays
lat_vals = lat.values  # shape (j,)
lon_vals = lon.values  # shape (i,)

# Create 2D lat/lon meshgrid
lon2d, lat2d = np.meshgrid(lon_vals, lat_vals, indexing='xy')  # shape (j, i)

######### Load surface forcings #########

SIarea = ds1.SIarea.isel(time=time_inst,face=face,i=i,j=j)     # SEAICE fractional ice-covered area [0 to 1]
SIhsalt = ds1.SIhsalt.isel(time=time_inst,face=face,i=i,j=j)   # SEAICE effective salinity
SIheff = ds1.SIheff.isel(time=time_inst,face=face,i=i,j=j)     # SEAICE effective ice thickness
SIhsnow = ds1.SIhsnow.isel(time=time_inst,face=face,i=i,j=j)   # SEAICE effective snow thickness
SIuice = ds1.SIuice.isel(time=time_inst,face=face,i_g=i_g,j=j) # SEAICE zonal ice velocity, >0 from West to East
SIvice = ds1.SIvice.isel(time=time_inst,face=face,i=i,j_g=j_g) # SEAICE merid. ice velocity, >0 from South to North

oceQsw = ds1.oceQsw.isel(time=time_inst,face=face,i=i,j=j)     # net Short-Wave radiation (+=down), >0 increases theta
oceFWflx = ds1.oceFWflx.isel(time=time_inst,face=face,i=i,j=j) # net surface Fresh-Water flux into the ocean (+=down), >0 decreases salinity
oceSflux = ds1.oceSflux.isel(time=time_inst,face=face,i=i,j=j) # net surface Salt flux into the ocean (+=down), >0 increases salinity
oceQnet = ds1.oceQnet.isel(time=time_inst,face=face,i=i,j=j)   # net surface heat flux into the ocean (+=down), >0 increases theta
oceTAUX = ds1.oceTAUX.isel(time=time_inst,face=face,i_g=i_g,j=j)  # zonal surface wind stress, >0 increases uVel
oceTAUY = ds1.oceTAUY.isel(time=time_inst,face=face,i=i,j_g=j_g)  # meridional surf. wind stress, >0 increases vVel


# plt.figure(figsize=(9, 6))
# pcm = plt.pcolormesh(lon2d, lat2d, oceQsw, cmap='twilight_shifted', shading='auto')
# plt.colorbar(pcm, label="oceQsw")
# plt.title("Shortwave Radiation")
# plt.xlabel("Longitude")
# plt.ylabel("Latitude")
# plt.grid(True, linestyle=':')
# plt.tight_layout()
# plt.savefig(f"{figdir}/oceQsw.png", dpi=150)
# plt.close()


variables = [
    (SIarea, "Sea Ice Area Fraction", "[0–1]"),
    (SIhsalt, "Sea Ice Effective Salinity", "(psu)"),
    (SIheff, "Sea Ice Thickness", "(m)"),
    (SIhsnow, "Sea Ice Snow Thickness", "(m)"),
    (SIuice, "Sea Ice Zonal Velocity", "(m/s)"),
    (SIvice, "Sea Ice Meridional Velocity", "(m/s)"),
    (oceTAUX, "Zonal Wind Stress", "(N/m²)"),
    (oceTAUY, "Meridional Wind Stress", "(N/m²)"),
    (oceQnet, "Net Heat Flux", "(W/m²)"),
    (oceQsw, "Shortwave Radiation", "(W/m²)"),
    (oceFWflx, "Freshwater Flux", "(kg/m²/s)"),
    (oceSflux, "Salt Flux", "(kg/m²/s)"),
]

for fig_idx in range(2):
    fig, axs = plt.subplots(3, 2, figsize=(14, 12))
    axs = axs.flatten()

    for i in range(6):
        var_idx = fig_idx * 6 + i
        var, title, unit = variables[var_idx]
        ax = axs[i]

        data = var.values
        data_flat = data[~np.isnan(data)]
        vmin, vmax = np.percentile(data_flat, [1, 99])

        im = ax.pcolormesh(lon2d, lat2d, var, shading='auto', cmap='coolwarm', vmin=vmin, vmax=vmax)
        ax.set_title(f"{title}")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        cb = fig.colorbar(im, ax=ax, orientation='vertical', label=unit)
        cb.formatter = ticker.ScalarFormatter(useMathText=True)
        cb.formatter.set_powerlimits((-2, 2))
        cb.update_ticks()

    plt.tight_layout()
    plt.savefig(f"{figdir}/surface_forcing_{fig_idx+1}.png", dpi=150)
    plt.close()
