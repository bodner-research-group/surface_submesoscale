##### Time series of the following variables:
#####
##### Qnet (net surface heat flux into the ocean),
##### Hml (mixed-layer depth), 
##### TuH (horizontal Turner angle), 
##### TuV (vertical Turner angle),
##### wb_cros (variance-perserving cross-spectrum of vertical velocity and buoyancy), 
##### Lmax (the horizontal length scale corresponds to wb_cros minimum), 
##### Dmax (the depth corresponds to wb_cros minimum), 
##### gradSSH (absolute gradient of sea surface height anomaly), etc.
#####
##### Step 1: compute 12-hour averages of temperature, salinity, and vertical velocity, save as .nc files


# Imports
import xarray as xr
import numpy as np
import os
import time
from dask.distributed import Client, LocalCluster

# ========== Dask cluster setup ==========
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Paths ==========
ds_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
output_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin"
os.makedirs(output_dir, exist_ok=True)

# ========== Domain and chunking ==========
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)
chunk_dict = {'time': 12, 'i': 120, 'j': 120}

# ========== Time settings ==========
nday_avg = 364
delta_days = 7
start_hours = 49 * 24
end_hours = start_hours + 24 * nday_avg
step_hours = delta_days * 24

# ========== Open dataset ==========
print("Opening dataset...")
ds1 = xr.open_zarr(ds_path, consolidated=False)

# ========== Function to process and save ==========
def compute_and_save_weekly(var_name, label):
    print(f"\n Starting: {label} (variable: {var_name})")

    total_segments = (end_hours - start_hours) // step_hours
    for n in range(total_segments):
        t0 = start_hours + n * step_hours
        t1 = t0 + step_hours

        t_start_wall = time.time()

        # Slice & chunk
        da = ds1[var_name].isel(time=slice(t0, t1), face=face, i=i, j=j).chunk(chunk_dict)

        # Compute 12h mean
        da_12h = da.coarsen(time=12, boundary='trim').mean()
        da_12h = da_12h.compute()

        # File name with datetime
        date_str = str(ds1.time.isel(time=t0).values)[:10]
        outname = os.path.join(output_dir, f"{label}_12h_{date_str}.nc")
        da_12h.to_netcdf(outname)

        print(f"   Week {n+1}/{total_segments} done in {(time.time() - t_start_wall)/60:.2f} min → {outname}")

        # Cleanup
        del da, da_12h

# ========== Main ==========
if __name__ == "__main__":
    variable_map = {
        "Theta": "tt",
        "Salt": "ss",
        "W": "ww"
    }

    for var_name, label in variable_map.items():
        compute_and_save_weekly(var_name, label)

    print("\n All variables processed and saved weekly.")