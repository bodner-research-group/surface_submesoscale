### Find the i, j, and face indices for the selected region.

### The output from this script is:
# ✅ Found matching grid region(s) for the selected region:
# Face 1:
#   i-range: 1824 to 4319
#   j-range: 179 to 1651
#   Number of points: 3676608

# Face 4:
#   i-range: 0 to 383
#   j-range: 179 to 1651
#   Number of points: 565632


import xarray as xr
import numpy as np

# Open the grid data (you've already done this)
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320', consolidated=False)

# Coordinates
XC = ds1['XC']  # longitude: dimensions (face, j, i)
YC = ds1['YC']  # latitude: dimensions (face, j, i)

# Define bounding box for the selected site
# Southern Ocean
lat_min, lat_max = -55, -35
lon_min, lon_max = 0, 60 # West longitudes as negative

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

