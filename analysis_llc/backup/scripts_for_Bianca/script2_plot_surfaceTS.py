### Plot surface temperature and salinity to verify that the corrected region is properly selected.

import xarray as xr
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt

# Open dataset
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320', consolidated=False)

# Define region info (replace these with your actual results)

# # ========== Domain ==========
domain_name = "US_West_Coast"
face = 10
i = slice(2765,3217,1) 
j = slice(0,287,1)

k = 0        # vertical index for ocean surface  (k can be 0~50)
t = 0        # first time step (2011-09-13 at 00:00)

i_min = i.start
i_max = i.stop
j_min = j.start
j_max = j.stop

### Path to the folder where figures will be saved (replace this with your own folder)
figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/"
os.makedirs(figdir, exist_ok=True)

# Extract subregion for T and S
theta = ds1['Theta'].isel(face=face, i=i, j=j, k=k, time=t)
salt  = ds1['Salt'].isel(face=face, i=i, j=j, k=k, time=t)
lon = ds1['XC'].isel(face=face, i=i, j=j)
lat = ds1['YC'].isel(face=face, i=i, j=j)

# Plot function
def plot_station_papa_region(var, lon, lat, title, cmap, vmin=None, vmax=None, filename='output.png'):
    plt.figure(figsize=(8, 6))
    plt.pcolormesh(lon, lat, var, cmap=cmap, shading='auto', vmin=vmin, vmax=vmax)
    plt.colorbar(label=title)
    plt.title(title + f"\n(face {face}, i={i_min}-{i_max}, j={j_min}-{j_max})")
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()
    print(f"✅ Saved plot: {filename}")

# Plot surface temperature
plot_station_papa_region(
    var=theta,
    lon=lon,
    lat=lat,
    title="Surface Temperature of the Selected region",
    cmap="coolwarm",
    vmin=theta.min().values,
    vmax=theta.max().values,
    filename=f"{figdir}selected_surface_theta.png"
)

# Plot surface salinity
plot_station_papa_region(
    var=salt,
    lon=lon,
    lat=lat,
    title="Surface Salinity of the Selected region",
    cmap="viridis",
    vmin=salt.min().values,
    vmax=salt.max().values,
    filename=f"{figdir}selected_surface_salt.png"
)


