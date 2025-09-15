import os
import glob
import numpy as np
import xarray as xr
from scipy import interpolate
from inpoly.inpoly2 import inpoly2
import pyinterp
import warnings
from datetime import datetime, timedelta
import pandas as pd

# === User inputs ===
cycle_dir = "cycle_008"
swot_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/data_swot/global_swot_grid_2024/" + cycle_dir
model_file = "/orcd/data/abodner/003/LLC4320/LLC4320"
output_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/data_llc/llc4320_to_swot/" + cycle_dir
interpolator = "pyinterp_interpolator"  # or "scipy_interpolator"
model_lat_var = "YC"
model_lon_var = "XC"
model_time_var = "time"
model_ssh_var = "Eta"

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Load model metadata
print("Loading model dataset...")
ds_model_all = xr.open_zarr(model_file, consolidated=False)

model_times = ds_model_all[model_time_var].values  # time coordinates from LLC4320
# model_times = np.array([np.datetime64(t).astype('datetime64[h]') for t in model_times])  # hourly resolution

# Convert model_times to pandas.DatetimeIndex for easy handling
model_times = pd.to_datetime(model_times)

# Get all SWOT files in directory
swot_files = sorted(glob.glob(os.path.join(swot_dir, "SWOT_GRID_L3_LR_SSH_*.nc")))

print(f"Found {len(swot_files)} SWOT files.")

for swot_file in swot_files:
    # === Extract time from filename ===
    fname = os.path.basename(swot_file)
    try:
        time_str = fname.split("_")[-3]
        time_start = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
        time_str = fname.split("_")[-2].split("_v")[0]
        time_end = datetime.strptime(time_str, "%Y%m%dT%H%M%S")
    except Exception as e:
        print(f"Skipping file due to time parse error: {fname}")
        continue

    mean_time = time_start + (time_end - time_start) / 2
    mean_time_np = np.datetime64(mean_time.replace(minute=0, second=0))  # Round to nearest hour

    # Convert mean_time to pandas.Timestamp
    mean_time = pd.to_datetime(mean_time_np)

    # Extract month-day-hour pattern
    mask_same_time = (
        (model_times.month == mean_time.month) &
        (model_times.day == mean_time.day) &
        (model_times.hour == mean_time.hour)
    )

    # Split by year
    times_2011 = model_times[(model_times.year == 2011) & mask_same_time]
    times_2012 = model_times[(model_times.year == 2012) & mask_same_time]

    # print("Matches in 2011:", times_2011)
    # print("Matches in 2012:", times_2012)

    model_timestep_index = None  

    if len(times_2011) > 0:
        model_timestep_index = np.where(model_times == times_2011[0])[0][0]
    elif len(times_2012) > 0:
        model_timestep_index = np.where(model_times == times_2012[0])[0][0]

    print(f"\nProcessing file: {fname}")
    print(f"Mean time: {mean_time} → Closest model time: {model_times[model_timestep_index]} (index {model_timestep_index})")

    # === Load SWOT data ===
    ds_swot = xr.open_dataset(swot_file, engine="netcdf4")

    # === Subset model at closest timestep ===
    ds_model = ds_model_all.isel({model_time_var: model_timestep_index})

    # === Prepare model data ===
    var_values = np.concatenate(ds_model[model_ssh_var].values, axis=1)
    lat_values = np.concatenate(ds_model[model_lat_var].values, axis=1)
    lon_values = np.concatenate(ds_model[model_lon_var].values, axis=1)

    # Clean model grid points
    mask_valid = ~((lon_values.flatten() == 0) & (lat_values.flatten() == 0))
    lon_clean = lon_values.flatten()[mask_valid]
    lat_clean = lat_values.flatten()[mask_valid]
    var_clean = var_values.flatten()[mask_valid]
    points = np.column_stack((lon_clean, lat_clean))

    # === Create polygon from SWOT swath ===
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
        continue

    # === Interpolation ===
    if interpolator == "scipy_interpolator":
        finterp = interpolate.LinearNDInterpolator(list(zip(lat_in, lon_in)), var_in, fill_value=np.nan)
    elif interpolator == "pyinterp_interpolator":
        points_for_pyinterp = np.column_stack((lon_in, lat_in))
        finterp = pyinterp.RTree()
        finterp.packing(points_for_pyinterp, var_in)
    else:
        raise ValueError(f"Unknown interpolator: {interpolator}")

    # === Interpolate to SWOT grid ===
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

    # === Build output dataset ===
    ds_out = xr.Dataset({
        "ssh": (["num_lines", "num_pixels"], ssh_interp)
    }, coords={
        "latitude": (["num_lines", "num_pixels"], lat_swot),
        "longitude": (["num_lines", "num_pixels"], ds_swot.longitude.values)
    })

    # Apply SWOT quality mask
    cross_dist = ds_swot.cross_track_distance
    quality_flag = ds_swot.quality_flag
    mask_dist = xr.where(
        (abs(cross_dist) <= 60.0) & (abs(cross_dist) >= 10.0) & (quality_flag < 101),
        cross_dist,
        np.nan
    )
    ds_out["ssh"] = ds_out["ssh"].where(~np.isnan(mask_dist))

    # Convert longitude to 0–360
    ds_out.coords['longitude'] = xr.where(ds_out.longitude < 0, ds_out.longitude + 360, ds_out.longitude)

    # === Save output ===
    out_fname = "llc2swot_" + mean_time.strftime("%Y%m%dT%H") + ".nc"
    out_path = os.path.join(output_dir, out_fname)
    ds_out.to_netcdf(out_path)

    print(f"Saved: {out_path}")

print("\nAll files processed.")
