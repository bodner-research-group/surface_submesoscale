import xarray as xr
import os
import numpy as np
from dask.distributed import Client, LocalCluster
from datetime import datetime

# ========== Start Dask Cluster ==========
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit='5.5GB')
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Paths ==========
input_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/TSW_12h_avg"
output_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/TSW_24h_avg"
os.makedirs(output_dir, exist_ok=True)

# ========== Variables ==========
variables = ["tt", "ss", "ww"]

# ========== Get files ==========
def get_files_for_var(var_label):
    files = sorted([
        os.path.join(input_dir, f)
        for f in os.listdir(input_dir)
        if f.startswith(var_label + "_") and f.endswith(".nc")
    ])
    return files

# ========== Main processing function ==========
def convert_12h_to_24h_weekly(var_label):
    print(f"\nProcessing variable: {var_label}")

    files_12h = get_files_for_var(var_label)
    if not files_12h:
        print(f"  No files found for {var_label}")
        return

    # Open with Dask
    ds_12h = xr.open_mfdataset(
        files_12h,
        combine="by_coords",
        parallel=True,
        chunks={"time": 24}
    ).sortby("time")

    # Time check
    time_diff = (ds_12h.time[1] - ds_12h.time[0]).values / np.timedelta64(1, 'h')
    print(f"  Time step: {time_diff} hours")
    if time_diff != 12:
        print("  Warning: Expected 12-hour steps!")

    # Total number of weeks
    n_time = ds_12h.sizes["time"]
    n_weeks = (n_time // 2) // 7  # each week = 14 timesteps (2 per day)

    print(f"  Total weeks to process: {n_weeks}")

    for week in range(n_weeks):
        t0 = week * 14
        t1 = t0 + 14

        # Slice one week (14 time steps = 7 days)
        week_ds = ds_12h.isel(time=slice(t0, t1))

        # Coarsen into 7 x 24h steps (14 x 12h)
        ds_24h = week_ds.coarsen(time=2, boundary="trim").mean()

        # Compute now (parallel)
        ds_24h = ds_24h.compute()

        # Get date string from the first time step
        date_str = str(ds_24h.time[0].values)[:10]

        # Save weekly file
        outname = os.path.join(output_dir, f"{var_label}_24h_{date_str}.nc")
        ds_24h.to_netcdf(outname)
        print(f"  Week {week+1}/{n_weeks} saved to {outname}")

# ========== Run ==========
if __name__ == "__main__":
    for var in variables:
        convert_12h_to_24h_weekly(var)

    print("\nAll variables processed and saved weekly.")
