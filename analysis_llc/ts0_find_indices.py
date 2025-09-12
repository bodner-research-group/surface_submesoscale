import xarray as xr
import numpy as np

# Open the grid data (you've already done this)
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320', consolidated=False)

# Coordinates
XC = ds1['XC']  # longitude: dimensions (face, j, i)
YC = ds1['YC']  # latitude: dimensions (face, j, i)

# Define bounding box for selected site

# # Ocean Weather Station Papa 
# lat_min, lat_max = 48.0, 52.0
# lon_min, lon_max = -147.0, -143.0  # West longitudes as negative

# # SWOT Validation Site 
# lat_min, lat_max = 33.5, 37.5
# lon_min, lon_max = -127.0, -123.0  # West longitudes as negative

# Antarctic Peninsula  
lat_min, lat_max = -63, -59
lon_min, lon_max = -62, -50 # West longitudes as negative


# Loop through all faces to find which face(s) cover the region
results = []

for face in range(XC.sizes['face']):
    lon_face = XC.isel(face=face).values
    lat_face = YC.isel(face=face).values

    # Boolean mask for points inside the bounding box
    mask = (
        (lat_face >= lat_min) & (lat_face <= lat_max) &
        (lon_face >= lon_min) & (lon_face <= lon_max)
    )

    if np.any(mask):
        j_idx, i_idx = np.where(mask)
        i_min, i_max = i_idx.min(), i_idx.max()
        j_min, j_max = j_idx.min(), j_idx.max()

        results.append({
            'face': face,
            'i_range': (i_min, i_max),
            'j_range': (j_min, j_max),
            'num_points': mask.sum()
        })

# Output
if results:
    print("✅ Found matching grid region(s) for selected region:")
    for r in results:
        print(f"Face {r['face']}:")
        print(f"  i-range: {r['i_range'][0]} to {r['i_range'][1]}")
        print(f"  j-range: {r['j_range'][0]} to {r['j_range'][1]}")
        print(f"  Number of points: {r['num_points']}\n")
else:
    print("❌ No matching region found.")







### Plot surface T, S 

import xarray as xr
import matplotlib.pyplot as plt
import os
import matplotlib.pyplot as plt


# Open dataset (already done)
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320', consolidated=False)

# Define region info (replace these with your actual results)

# # ========== Domain ==========
# domain_name = "Station_Papa"
# face = 7
# i = slice(1873,2189+1,1)  # Ocean Weather Station Papa  147W-143W  
# j = slice(3408,3599+1,1) # Ocean Weather Station Papa  48N-52N

# ========== Domain ==========
domain_name = "SWOT_Site"
face = 10
i = slice(0,480,1)  # SWOT Validation Site 126W-124W
j = slice(2931-113,3186+113,1) # SWOT Validation Site  35N-37N
# i = slice(48,239,1)  # SWOT Validation Site 126W-124W
# j = slice(2931,3186,1) # SWOT Validation Site  35N-37N


# # ========== Domain ==========
# domain_name = "Antarctic_Peninsula"
# face = 12
# i = slice(183-40,586+40,1) 
# j = slice(3168,3743,1)

k = 0        # surface
t = 0        # first time step


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
    title="Surface Temperature at Antarctic Peninsula",
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
    title="Surface Salinity at Antarctic Peninsula",
    cmap="viridis",
    vmin=salt.min().values,
    vmax=salt.max().values,
    filename=f"{figdir}selected_surface_salt.png"
)

