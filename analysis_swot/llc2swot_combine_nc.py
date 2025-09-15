import os
import glob
import numpy as np
import xarray as xr
import pandas as pd
from datetime import datetime

# Base path for input and output folders
base_input_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/data_swot/llc4320_to_swot"
base_output_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/data_swot/llc4320_to_swot_combined"

# Path to your model dataset with time variable to find model_times
model_ds_path = "/orcd/data/abodner/003/LLC4320/LLC4320"

print("Loading model dataset times...")
ds_model_all = xr.open_zarr(model_ds_path, consolidated=False)
model_times = pd.to_datetime(ds_model_all["time"].values)

def extract_swot_time(fname):
    parts = fname.split("_")
    try:
        time_start_str = parts[-3]
        time_end_str = parts[-2]
        time_end_str = time_end_str.split("_v")[0] if "_v" in time_end_str else time_end_str
        dt_start = datetime.strptime(time_start_str, "%Y%m%dT%H%M%S")
        dt_end = datetime.strptime(time_end_str, "%Y%m%dT%H%M%S")
        return dt_start, dt_end
    except Exception as e:
        print(f"Failed to parse times from filename {fname}: {e}")
        return None, None

def find_model_timestep_index(swot_dt_start, swot_dt_end, model_times):
    mean_time = swot_dt_start + (swot_dt_end - swot_dt_start) / 2
    mean_time_rounded = mean_time.replace(minute=0, second=0, microsecond=0)

    mask_same_time = (
        (model_times.month == mean_time_rounded.month) &
        (model_times.day == mean_time_rounded.day) &
        (model_times.hour == mean_time_rounded.hour)
    )

    times_2011 = model_times[(model_times.year == 2011) & mask_same_time]
    times_2012 = model_times[(model_times.year == 2012) & mask_same_time]

    model_timestep_index = None
    if len(times_2011) > 0:
        model_timestep_index = np.where(model_times == times_2011[0])[0][0]
    elif len(times_2012) > 0:
        model_timestep_index = np.where(model_times == times_2012[0])[0][0]

    return model_timestep_index

for cycle_num in range(15, 16):
    cycle_str = f"cycle_{cycle_num:03d}"
    print(f"\nProcessing {cycle_str} ...")

    cycle_dir = os.path.join(base_input_dir, cycle_str)

    nc_files = sorted(glob.glob(os.path.join(cycle_dir, "*.nc")))

    if not nc_files:
        print(f"No .nc files found in {cycle_dir}, skipping...")
        continue

    print(f"Found {len(nc_files)} files in {cycle_dir}")

    datasets = []

    for nc_file in nc_files:
        fname = os.path.basename(nc_file)
        dt_start, dt_end = extract_swot_time(fname)
        if dt_start is None or dt_end is None:
            print(f"Skipping file due to time parse error: {fname}")
            continue

        model_idx = find_model_timestep_index(dt_start, dt_end, model_times)
        if model_idx is None:
            print(f"No matching model time found for file: {fname}")
            continue

        ds = xr.open_dataset(nc_file)

        mean_swot_time = dt_start + (dt_end - dt_start) / 2

        ds = ds.expand_dims("file")  # new dimension for concat

        ds = ds.assign_coords({
            "swot_time": pd.Timestamp(mean_swot_time),
            "model_time": model_times[model_idx]
        })

        datasets.append(ds)

    if not datasets:
        print(f"No valid datasets to merge for {cycle_str}, skipping save.")
        continue

    print("Merging datasets...")
    combined_ds = xr.concat(datasets, dim="file")

    output_file = os.path.join(base_output_dir, f"LLC4320_on_SWOT_GRID_L3_LR_SSH_{cycle_num:03d}_combined.nc")
    combined_ds.to_netcdf(output_file)
    print(f"Saved combined dataset to {output_file}")

print("\nAll cycles processed.")
