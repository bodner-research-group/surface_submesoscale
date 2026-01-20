import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import os
import matplotlib.colors as colors
from set_colormaps import WhiteBlueGreenYellowRed

# from set_constant import domain_name, face, i, j, start_hours, end_hours, step_hours

# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

# i = slice(560, 980)   # smaller domain for plotting 
# j = slice(3000, 3400)  # smaller domain for plotting 
# i_g = slice(560, 981)   # smaller domain for plotting 
# j_g = slice(3000, 3401)  # smaller domain for plotting 

cmap = WhiteBlueGreenYellowRed()

# ==== Constants ====
omega = 7.2921e-5  # [rad/s]

# Global font size setting for figures
plt.rcParams.update({'font.size': 17})

# ==== Paths ====
data_path = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/strain_vorticity/strain_vorticity_daily.nc"
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/Manuscript_Data/{domain_name}/"
os.makedirs(figdir, exist_ok=True)

# ==== Load data ====
ds = xr.open_dataset(data_path)

# Load the model
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320',consolidated=False)

# Coordinate
lat = ds1.YC.isel(face=face,i=i,j=j)
lon = ds1.XC.isel(face=face,i=i,j=j)
lat_g = ds1.YG.isel(face=face,i_g=i,j_g=j)
lon_g = ds1.XG.isel(face=face,i_g=i,j_g=j)
# lat_g = ds1.YG.isel(face=face,i_g=i_g,j_g=j_g)
# lon_g = ds1.XG.isel(face=face,i_g=i_g,j_g=j_g)

# ==== Compute Coriolis parameter (mean) ====
lat_rad = np.deg2rad(lat)
f0 = 2 * omega * np.sin(lat_rad)
f0_mean = f0.mean().compute()
print(f"Mean f0 over domain: {f0_mean:.2e} s^-1")

# ==== Normalize variables ====
sigma_norm = ds["strain_mag"] / abs(f0_mean)
zeta_norm = ds["vorticity"] / f0_mean
delta_norm = ds["divergence"] / f0_mean

# ==== Daily Maps ====
print("\nGenerating daily maps...")
# for t in range(len(ds.time)):
# t = 133
t = 300

date_str = str(ds.time[t].values)[:10]

# Create figure with 3 subplots
fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), constrained_layout=True)

# Data for current time step
v0 = zeta_norm.isel(time=t).sel(i_g=i,j_g=j)
v1 = sigma_norm.isel(time=t).sel(i=i,j=j)
v2 = delta_norm.isel(time=t).sel(i=i,j=j)

# Add date as figure-level title
fig.suptitle(f"Date: {date_str}", fontsize=20)

# subplot 1
im0 = axes[0].pcolormesh(lon_g, lat_g, v0, cmap='RdBu_r', shading="auto", vmin=-1, vmax=1)
axes[0].set_title(r'$\zeta/f_0$')
axes[0].set_xlabel("Longitude")
axes[0].set_ylabel("Latitude")
cbar0 = plt.colorbar(im0, ax=axes[0], orientation="vertical", pad=0.02, fraction=0.032, shrink=0.85)
# cbar0.set_label(r'$\zeta/f_0$')

# subplot 2
im1 = axes[1].pcolormesh(lon_g, lat_g, v1, cmap='viridis', shading="auto", vmin=0, vmax=1)
axes[1].set_title(r'$\sigma/|f_0|$')
axes[1].set_xlabel("Longitude")
# axes[1].set_ylabel("Latitude")
cbar1 = plt.colorbar(im1, ax=axes[1], orientation="vertical", pad=0.02, fraction=0.032, shrink=0.85)
# cbar1.set_label(r'$\sigma/|f_0|$')

# subplot 3
im2 = axes[2].pcolormesh(lon_g, lat_g, v2, cmap='BrBG', shading="auto", vmin=-0.5, vmax=0.5)
axes[2].set_title(r'$\Delta/f_0$')
axes[2].set_xlabel("Longitude")
# axes[2].set_ylabel("Latitude")
cbar2 = plt.colorbar(im2, ax=axes[2], orientation="vertical", pad=0.02, fraction=0.032, shrink=0.85)
# cbar2.set_label(r'$\Delta/f_0$')


lon_min, lon_max = -27.0, -17.5
lat_min, lat_max = 57.9, 62.3

for ax in axes:
    ax.set_xlim(lon_min, lon_max)
    ax.set_ylim(lat_min, lat_max)


# Save figure
outpath = os.path.join(figdir, f"combined_norm_map_{date_str}.png")
plt.savefig(outpath, dpi=300)
plt.close()
