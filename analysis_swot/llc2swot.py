# Imports
import xarray as xr
import numpy as np
import pyinterp
import os
from scipy import interpolate
from inpoly.inpoly2 import inpoly2
import warnings
from xmitgcm import llcreader  # only import if you have LLC4320 data

# === User inputs (replace with your paths/parameters) ===
model_file = "/orcd/data/abodner/003/LLC4320/LLC4320"
swot_file = "/orcd/data/abodner/002/ysi/surface_submesoscale/data_swot/global_swot_grid_2024/cycle_010/SWOT_GRID_L3_LR_SSH_010_295_20240204T122458_20240204T131624_v1.0.2.nc"
output_file = "/orcd/data/abodner/002/ysi/surface_submesoscale/data_llc/test.nc"
interpolator = "pyinterp_interpolator"  # or "scipy_interpolator"

model_lat_var = "YC"
model_lon_var = "XC"
model_time_var = "time"
model_ssh_var = "Eta"
model_timestep_index = 1
# model_face = 6

# === Read model and mask datasets ===
ds_model =  xr.open_zarr(model_file, consolidated=False)

# Read SWOT dataset with netcdf4 engine
ds_swot = xr.open_dataset(swot_file, engine="netcdf4")

# Select first timestep in model if time dimension exists
ds_model = ds_model.isel({model_time_var: model_timestep_index})

# === Prepare model variable for interpolation ===
# var_values = ds_model.Eta.isel(face=model_face).values
# lat_values = ds_model.YC.isel(face=model_face).values
# lon_values = ds_model.XC.isel(face=model_face).values

var_values = np.concatenate(ds_model.Eta.values, axis=1) 
lat_values = np.concatenate(ds_model.YC.values, axis=1) 
lon_values = np.concatenate(ds_model.XC.values, axis=1) 


# === Select model data inside SWOT swath ===
# Clean model grid points (remove (0,0))
mask_valid = ~((lon_values.flatten() == 0) & (lat_values.flatten() == 0))
lon_clean = lon_values.flatten()[mask_valid]
lat_clean = lat_values.flatten()[mask_valid]
var_clean = var_values.flatten()[mask_valid]

points = np.column_stack((lon_clean, lat_clean))

# Adjust SWOT longitude from 0-360 to -180/180
X = xr.where(ds_swot.longitude <= 180, ds_swot.longitude, ds_swot.longitude - 360)
Y = ds_swot.latitude

dy = Y.values[-1, 0] - Y.values[0, 0]
k = 2 if dy > 0 else -2
k1 = abs(k)

xx = np.concatenate([
    X.isel(num_lines=0).values,
    X.isel(num_pixels=-1).values + k,
    X.isel(num_lines=-1).values[::-1],
    X.isel(num_pixels=0).values[::-1] - k
])

yy = np.concatenate([
    Y.isel(num_lines=0).values - k,
    Y.isel(num_pixels=-1).values - k1,
    Y.isel(num_lines=-1).values[::-1] + k,
    Y.isel(num_pixels=0).values[::-1] + k1
])

polygon = np.column_stack((xx, yy))
inside, on_edge = inpoly2(points, polygon)
mask = inside | on_edge

lon_in = lon_clean[mask]
lat_in = lat_clean[mask]
var_in = var_clean[mask]

if np.size(var_in) == 0:
    warnings.warn("No model data found within SWOT swath area.")
    finterp = 0
else:
    # === Build interpolator ===
    if interpolator == "scipy_interpolator":
        finterp = interpolate.LinearNDInterpolator(list(zip(lat_in, lon_in)), var_in, fill_value=np.nan)
    elif interpolator == "pyinterp_interpolator":
        points_for_pyinterp = np.column_stack((lon_in, lat_in))
        finterp = pyinterp.RTree()
        finterp.packing(points_for_pyinterp, var_in)
    else:
        raise ValueError(f"Unknown interpolator: {interpolator}")

# === Interpolate satellite data ===
if finterp != 0:
    # Convert SWOT longitude back to -180/180
    lat_swot = ds_swot.latitude.values
    lon_swot = xr.where(ds_swot.longitude > 180, ds_swot.longitude - 360, ds_swot.longitude).values

    if interpolator == "scipy_interpolator":
        points_swot = np.column_stack((lat_swot.flatten(), lon_swot.flatten()))
        ssh_interp = finterp(points_swot).reshape(lat_swot.shape)
    elif interpolator == "pyinterp_interpolator":
        points_swot = np.column_stack((lon_swot.flatten(), lat_swot.flatten()))
        ssh_interp = finterp.inverse_distance_weighting(
            coordinates=points_swot,
            k=5,
            num_threads=0,
            p=2,
            within=True
        )[0].reshape(lat_swot.shape)

    # Build output xarray dataset
    var_name_out = "ssh"
    ds_out = xr.Dataset({
        var_name_out: (["num_lines", "num_pixels"], ssh_interp)
    }, coords={
        "latitude": (["num_lines", "num_pixels"], lat_swot),
        "longitude": (["num_lines", "num_pixels"], ds_swot.longitude.values)
    })

    # Apply quality mask (valid distance 10-60 km, quality_flag < 101)
    cross_dist = ds_swot.cross_track_distance
    quality_flag = ds_swot.quality_flag

    mask_dist = xr.where(
        (abs(cross_dist) <= 60.0) & (abs(cross_dist) >= 10.0) & (quality_flag < 101),
        cross_dist,
        np.nan
    )
    ds_out["ssh"] = ds_out["ssh"].where(~np.isnan(mask_dist))

    # Convert longitude back to 0-360
    ds_out.coords['longitude'] = xr.where(ds_out.longitude < 0, ds_out.longitude + 360, ds_out.longitude)

    # === Save to NetCDF ===
    ds_out.to_netcdf(output_file)

    print("Script finished successfully")
else:
    print("The model has no information for the SWOT path")
