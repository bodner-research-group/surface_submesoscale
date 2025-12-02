### Compute mixed-layer depth-averaged w, b, wb 

import xarray as xr
import numpy as np
import os
from dask.distributed import Client, LocalCluster
from glob import glob

from set_constant import domain_name, face, i, j

# ================= Directories =================
rho_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
w_dir   = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TSW_24h_avg"

output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_mld_daily"
os.makedirs(output_dir, exist_ok=True)

# ================= Load grid info =================
ds_grid = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

lat = ds_grid.YC.isel(face=face, i=i, j=j)
lon = ds_grid.XC.isel(face=face, i=i, j=j)
depth = ds_grid.Z              # k levels (51)
depth_kp1 = ds_grid.Zp1        # k+1 (52)
dz = ds_grid.drF               # vertical thickness (k)

depth3d, _, _ = xr.broadcast(depth, lon, lat)
dz3d, _, _ = xr.broadcast(dz, lon, lat)


# ================= Files =================
rho_files = sorted(glob(os.path.join(rho_dir, "rho_Hml_TS_daily_*.nc")))
hml_files = sorted(glob(os.path.join(rho_dir, "Hml_daily_surface_reference_*.nc")))
ww_files  = sorted(glob(os.path.join(w_dir,  "ww_24h_*.nc")))


# ================= Function for a single snapshot =================
def compute_mld_integrals_one_time(rho, Hml, w_kp1):

    gravity = 9.81
    rho0 = 1027.5

    # buoyancy
    b = -gravity * (rho - rho0) / rho0

    # interpolate W(k+1) → W(k)
    w_k = (
        w_kp1.rename({"k_p1": "Z"})
             .assign_coords(Z=depth_kp1.values)
             .interp(Z=depth)
             .fillna(0)
    )

    # mask depths above Hml (z=0 down to z=-Hml)
    mask = depth >= -Hml  # depth negative downward

    # integrals
    int_wb = (w_k * b * dz3d).where(mask).sum("Z")
    int_w  = (w_k * dz3d).where(mask).sum("Z")
    int_b  = (b * dz3d).where(mask).sum("Z")

    # depth-averaging
    wb_avg = int_wb / Hml
    w_avg  = int_w  / Hml
    b_avg  = int_b  / Hml

    # separated product
    wb_fact = w_avg * b_avg

    return wb_avg, w_avg, b_avg, wb_fact



# ================= Dask Cluster =================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)


# ================= Main Loop =================
for rho_file, Hml_file, ww_file in zip(rho_files, hml_files, ww_files):

    date_tag = os.path.basename(rho_file).split("_")[-1].replace(".nc", "")
    out_file = os.path.join(output_dir, f"wb_mld_daily_{date_tag}.nc")

    if os.path.exists(out_file):
        print(f"Skipping {date_tag} (exists)")
        continue

    print(f"Processing {date_tag} ...")


    # Load daily datasets (each has exactly 1 timestamp)
    rho = xr.open_dataset(rho_file)["rho_daily"].chunk({"i": -1, "j": -1})
    Hml = xr.open_dataset(Hml_file)["Hml_daily"].chunk({"i": -1, "j": -1})

    rho_time = rho.time.values

    # Load weekly W
    ww = xr.open_dataset(ww_file)["W"].chunk({"time": 1, "i": -1, "j": -1})

    # --- find the index of the W snapshot closest to rho_time ---
    ww_times = ww.time.values
    t_index = np.argmin(np.abs(ww_times - rho_time))

    print(f"ρ date = {rho_time}, using W time = {ww_times[t_index]}")

    # Extract the matched W snapshot
    w_matched = ww.isel(time=t_index)

    # Compute daily averages
    wb_avg, w_avg, b_avg, wb_fact = compute_mld_integrals_one_time(
        rho, Hml, w_matched
    )

    # Save output
    ds_out = xr.Dataset({
        "wb_avg": wb_avg,
        "w_avg":  w_avg,
        "b_avg":  b_avg,
        "wb_fact": wb_fact,
        "Hml": Hml
    })
    ds_out.to_netcdf(out_file)
    print(f"Saved → {out_file}")


client.close()
cluster.close()
