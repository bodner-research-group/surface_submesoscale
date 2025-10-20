### Compute potential density using the GSW tool box (non-linear equation of state)

# ========== Imports ==========
import xarray as xr
import numpy as np
import gsw
import os

# ========== Open dataset ==========
ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320', consolidated=False)

# ========== Domain ==========
domain_name = "US_West_Coast"
face = 10
i = slice(2765,3217,1) 
j = slice(0,287,1)

### Path to the folder where the NetCDF data will be saved (replace this with your own folder)
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}/"

# ========== Select time index ==========
t = 0        # first time step (2011-09-13 at 00:00)
###### LLC4320 has hourly output from 2011-09-13 at 00:00 to 2012-11-15 at 14:00

# ========== Select vertical index ==========
ds1.Z.values      ### print out the vertical coordinate
k = slice(0, 50, 1)  # Select data for the upper ~1000 m of the ocean. Adjust as needed. 
                     # The maximum k value is 50, corresponding to Z = -945.6 m at cell center.

# ========== Extract subregion for T and S ==========
theta = ds1['Theta'].isel(face=face, i=i, j=j, k=k, time=t)
salt  = ds1['Salt'].isel(face=face, i=i, j=j, k=k, time=t)
lon = ds1['XC'].isel(face=face, i=i, j=j)
lat = ds1['YC'].isel(face=face, i=i, j=j)
depth = ds1.Z.isel(k=k)                       # 1D vertical coordinate
depth3d, _, _ = xr.broadcast(depth, lon, lat) # Broadcast depth

# ========== Calculate SA and CT ==========
##### SA: Absolute Salinity, g/kg
SA = gsw.SA_from_SP(salt, depth3d, lon, lat)

##### CT: Conservative Temperature (ITS-90), degrees C
CT = gsw.CT_from_pt(SA, theta)

# ========== Calculate rho, alpha, beta ==========
p_ref = 0  ### Reference pressure

### Compute potential density rho, using a reference pressure of 0
rho = gsw.rho(SA, CT, p_ref)

# ### If you want to compute in-situ density, just replace p_ref with depth3d
# rho_insitu = gsw.rho (SA, CT, depth3d)



# ========== Define a function to compute mixed layer depth Hml ==========
def compute_Hml(rho_profile, depth_profile, threshold=0.03):
    rho_10m = rho_profile[6]  # density at ~10m depth
    mask = rho_profile > rho_10m + threshold
    if not np.any(mask):
        return 0.0
    return float(depth_profile[mask].max())

# ========== Compute Mixed Layer Depth ==========
Hml = xr.apply_ufunc(
    compute_Hml,
    rho,
    depth,
    input_core_dims=[["k"], ["k"]],
    output_core_dims=[[]],
    vectorize=True,
    dask="parallelized",
    output_dtypes=[float],
)


# ========== Save as NetCDF ==========
out_ds = xr.Dataset({
    "rho": rho,
    "Hml": Hml
})

out_path = os.path.join(output_dir, f"rho_Hml.nc")
out_ds.to_netcdf(out_path)
print(f"Saved: {out_path}")

