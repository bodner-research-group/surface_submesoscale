# Imports
import xarray as xr
import numpy as np
import os
import time
from dask.distributed import Client, LocalCluster
from dask import delayed, compute


# ========== Dask cluster setup ==========
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Paths ==========
ds_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
output_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin/surfaceUV_24h_avg"
os.makedirs(output_dir, exist_ok=True)

# ========== Domain ==========
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)

# ========== Time settings ==========
nday_avg = 364
delta_days = 7
start_hours = 49 * 24
end_hours = start_hours + 24 * nday_avg
step_hours = delta_days * 24

# ========== Open dataset without forcing chunk misalignment ==========
print("Opening dataset...")
ds1 = xr.open_zarr(ds_path, consolidated=False)

# ========== Task function ==========
@delayed
def process_and_save_week(t0, t1, var_name, label, i_name, j_name):
    t_wall = time.time()

    # Load & slice
    da = (
        ds1[var_name]
        .isel(time=slice(t0, t1), k=0, face=face, **{i_name: i, j_name: j})
        .chunk({'time': 24, i_name: 120, j_name: 120})  # optimal chunk
    )

    # Compute 24h mean
    da_24h = da.coarsen(time=24, boundary='trim').mean()

    # File name
    date_str = str(ds1.time.isel(time=t0).values)[:10]
    outname = os.path.join(output_dir, f"{label}_24h_{date_str}.nc")

    # Save
    da_24h.to_netcdf(outname)

    print(f"   Saved {label} week starting {date_str} in {(time.time() - t_wall)/60:.2f} min → {outname}")
    return outname

# ========== Main ==========
if __name__ == "__main__":
    variable_map = [
        # ("U", "uu_s", "i_g", "j"),
        ("V", "vv_s", "i", "j_g"),
    ]

    all_tasks = []

    for var_name, label, i_name, j_name in variable_map:
        print(f"\nStarting: {label} (variable: {var_name})")

        total_segments = (end_hours - start_hours) // step_hours
        for n in range(total_segments):
            t0 = start_hours + n * step_hours
            t1 = t0 + step_hours

            task = process_and_save_week(t0, t1, var_name, label, i_name, j_name)
            all_tasks.append(task)

    print(f"\nDispatching {len(all_tasks)} tasks to Dask...")
    results = compute(*all_tasks)  # trigger all
    print("\n All tasks completed.")



# # ========== Function to process and save ==========
# def compute_and_save_weekly(var_name, label, i_name, j_name):
#     print(f"\n Starting: {label} (variable: {var_name})")

#     total_segments = (end_hours - start_hours) // step_hours
#     chunk_dict = {'time': 24, i_name: 120, j_name: 120}
#     # chunk_dict = {'time': 24}

#     for n in range(total_segments):
#         t0 = start_hours + n * step_hours
#         t1 = t0 + step_hours

#         t_start_wall = time.time()

#         # Slice to surface and domain
#         da = (
#             ds1[var_name]
#             .isel(time=slice(t0, t1), k=0, face=face, **{i_name: i, j_name: j})
#             # .chunk(chunk_dict) 
#         )

#         # Compute 24h mean
#         da_24h = da.coarsen(time=24, boundary='trim').mean().compute()

#         # File name with datetime
#         date_str = str(ds1.time.isel(time=t0).values)[:10]
#         outname = os.path.join(output_dir, f"{label}_24h_{date_str}.nc")
#         da_24h.to_netcdf(outname)

#         print(f"   Week {n+1}/{total_segments} done in {(time.time() - t_start_wall)/60:.2f} min → {outname}")

#         # Cleanup
#         del da, da_24h

# # ========== Main ==========
# if __name__ == "__main__":
#     # var_name, label, i_coord, j_coord
#     variable_map = [
#         ("U", "uu_s", "i_g", "j"),
#         ("V", "vv_s", "i", "j_g"),
#     ]

#     for var_name, label, i_name, j_name in variable_map:
#         compute_and_save_weekly(var_name, label, i_name, j_name)

#     print("\n All variables processed and saved.")