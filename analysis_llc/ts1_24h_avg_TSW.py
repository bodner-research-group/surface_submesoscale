##### Compute 24-hour averages of temperature, salinity, and vertical velocity, save as .nc files

# Imports
import xarray as xr
import numpy as np
import os
import time
from dask.distributed import Client, LocalCluster

from set_constant import domain_name, face, i, j, start_hours, end_hours, step_hours

# ========== Paths ==========
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TSW_24h_avg"
os.makedirs(output_dir, exist_ok=True)

# ========== Open dataset ==========
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

# ========== Chunking ==========
# chunk_dict = {'time': 24, 'i': 120, 'j': 120}
chunk_dict = {'time': 24}


# ========== Function to process and save ==========
def compute_and_save_weekly(var_name, label):
    print(f"\n Starting: {label} (variable: {var_name})")

    total_segments = (end_hours - start_hours) // step_hours
    for n in range(total_segments):
        t0 = start_hours + n * step_hours
        t1 = t0 + step_hours

        # File name with datetime
        date_str = str(ds1.time.isel(time=t0).values)[:10]
        outname = os.path.join(output_dir, f"{label}_24h_{date_str}.nc")
        # Skip if file already exists
        if os.path.exists(outname):
            print(f"   Skipping week {n+1}/{total_segments} → already exists: {outname}")
            continue

        t_start_wall = time.time()

        # Slice & chunk
        da = ds1[var_name].isel(time=slice(t0, t1), face=face, i=i, j=j).chunk(chunk_dict)

        # Compute 24h mean
        da_24h = da.coarsen(time=24, boundary='trim').mean()
        da_24h = da_24h.compute()

        # File name with datetime
        date_str = str(ds1.time.isel(time=t0).values)[:10]
        outname = os.path.join(output_dir, f"{label}_24h_{date_str}.nc")
        da_24h.to_netcdf(outname)

        print(f"   Week {n+1}/{total_segments} done in {(time.time() - t_start_wall)/60:.2f} min → {outname}")

        # Cleanup
        del da, da_24h

# ========== Main ==========
if __name__ == "__main__":
    # ========== Dask cluster setup ==========
    cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    variable_map = {
        "Theta": "tt",
        "Salt": "ss",
        "W": "ww"
    }

    for var_name, label in variable_map.items():
        compute_and_save_weekly(var_name, label)

    print("\n All variables processed and saved weekly.")

    client.close()
    cluster.close()



