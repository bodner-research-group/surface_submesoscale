# ===== Imports =====
import os
import numpy as np
import xarray as xr
from datetime import timedelta
from dask.distributed import Client, LocalCluster
from set_constant import domain_name, face, i, j, start_hours, end_hours, step_hours

# =====================
# Setup Dask cluster
# =====================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# =====================
# Paths
# =====================
eta_input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/surface_24h_avg"
eta_weekly_out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/SSH_weekly"
os.makedirs(eta_weekly_out_dir, exist_ok=True)

# =====================
# Open reference dataset to get time axis
# =====================
# Load the main dataset just to get the time axis
zarr_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
ds_time = xr.open_zarr(zarr_path, consolidated=False)
time_ref = ds_time.time.isel(time=0).values  # starting time as numpy.datetime64

# Compute total segments
total_weeks = (end_hours - start_hours) // step_hours
print(f"Total weeks to process: {total_weeks}")

# =====================
# Weekly averaging loop
# =====================
for n in range(total_weeks):
    t0 = start_hours + n * step_hours
    t1 = t0 + step_hours

    # Get date for this week's starting day
    date_str = str((np.datetime64(time_ref) + np.timedelta64(t0, 'h')))[:10]
    out_path = os.path.join(eta_weekly_out_dir, f"eta_weekly_{date_str}.nc")

    if os.path.exists(out_path):
        print(f"  Skipping week {n+1}/{total_weeks} → already exists: {out_path}")
        continue

    print(f"  Processing week {n+1}/{total_weeks} ({date_str})")

    # Construct filename of 24h-averaged Eta for this week
    week_file = os.path.join(eta_input_dir, f"eta_24h_{date_str}.nc")

    if not os.path.exists(week_file):
        print(f"    WARNING: Missing file: {week_file} → skipping this week")
        continue

    # Open dataset using Dask
    ds_eta = xr.open_dataset(week_file, chunks={'time': 7, 'i': 100, 'j': 100})

    # Compute mean over the 7-day time axis
    eta_weekly = ds_eta['Eta'].mean(dim='time', keep_attrs=True).compute()
    eta_weekly = eta_weekly.rename('eta_7d')

    # Re-add time dimension with a single time value
    eta_weekly = eta_weekly.expand_dims("time")
    eta_weekly["time"] = [np.datetime64(time_ref) + np.timedelta64(t0, 'h')]

    # Save to netCDF
    eta_weekly.to_netcdf(out_path)
    print(f"    Saved to: {out_path}")

    # Cleanup
    ds_eta.close()
    del eta_weekly

# =====================
# Done
# =====================
print("\n✅ All weekly Eta files computed and saved.")
client.close()
cluster.close()
