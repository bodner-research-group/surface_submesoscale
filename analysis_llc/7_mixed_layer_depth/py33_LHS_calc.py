import xarray as xr
import numpy as np
import os
from glob import glob
from datetime import datetime, timedelta
from dask.distributed import Client, LocalCluster

from set_constant import domain_name, face, i, j

# ======================================================
# Dask setup
# ======================================================
cluster = LocalCluster(
    n_workers=64,
    threads_per_worker=1,
    memory_limit="5.5GB",
)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ======================================================
# Directories
# ======================================================
base_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"

rho_dir = f"{base_dir}/rho_Hml_TS_daily_avg"
rho_dir_td = f"{base_dir}/rho_Hml_TS_daily_avg_for_time_derivative"
output_dir = f"{base_dir}/LHS_integrated_buoyancy_tendency"

os.makedirs(output_dir, exist_ok=True)

# ======================================================
# Physical constants
# ======================================================
rho0 = 1027.5
g = 9.81
delta_rho = 0.03
dt = 86400.0  # seconds
min_H = 10.0  # minimum mixed-layer depth (m)

# ======================================================
# Load LLC grid (ONCE)
# ======================================================
grid_path = "/orcd/data/abodner/003/LLC4320/LLC4320"
ds_grid = xr.open_zarr(grid_path, consolidated=False)

lat = ds_grid.YC.isel(face=face, i=i, j=j)
lon = ds_grid.XC.isel(face=face, i=i, j=j)

depth = ds_grid.Z        # (k), negative downward
dz = ds_grid.drF         # (k)

# Broadcast dz to 3D for vertical integration
dz3d, _, _ = xr.broadcast(dz, lon, lat)

# ======================================================
# File lists
# ======================================================
rho_files = sorted(glob(f"{rho_dir}/rho_Hml_TS_daily_*.nc"))
hml_files = sorted(glob(f"{rho_dir}/Hml_daily_surface_reference_*.nc"))
rho_td_files = sorted(glob(f"{rho_dir_td}/rho_Hml_TS_daily_*.nc"))

def tag_from_file(fname):
    return os.path.basename(fname).split("_")[-1].replace(".nc", "")

rho_dict = {tag_from_file(f): f for f in rho_files}
hml_dict = {tag_from_file(f): f for f in hml_files}
rho_td_dict = {tag_from_file(f): f for f in rho_td_files}

date_tags = sorted(set(rho_dict) & set(hml_dict) & set(rho_td_dict))

# ======================================================
# Main loop
# ======================================================
for n, date_tag in enumerate(date_tags):

    current_date = datetime.strptime(date_tag, "%Y%m%d")
    prev_date = current_date - timedelta(days=1) if n == 0 else \
                datetime.strptime(date_tags[n - 1], "%Y%m%d")

    prev_tag = prev_date.strftime("%Y%m%d")

    out_file = f"{output_dir}/LHS_{date_tag}.nc"
    if os.path.exists(out_file):
        print(f"Skipping {date_tag} (exists)")
        continue

    print(f"Processing {date_tag} (prev={prev_tag})")

    # ==================================================
    # Load time-centered fields
    # ==================================================
    ds_rho = xr.open_dataset(rho_dict[date_tag]).chunk({"k": -1, "j": -1, "i": -1})
    rho = ds_rho["rho_daily"]          # (k,j,i)

    ds_hml = xr.open_dataset(hml_dict[date_tag])
    Hml = -ds_hml["Hml_daily"]         # positive depth (m)

    # ==================================================
    # Load staggered fields for time derivative
    # ==================================================
    ds_td = xr.open_dataset(rho_td_dict[date_tag]).chunk({"k": -1, "j": -1, "i": -1})
    ds_td_prev = xr.open_dataset(rho_td_dict[prev_tag]).chunk({"k": -1, "j": -1, "i": -1})

    rho_td = ds_td["rho_daily"]
    rho_td_prev = ds_td_prev["rho_daily"]

    Hml_td = -ds_td["Hml_daily"]
    Hml_td_prev = -ds_td_prev["Hml_daily"]

    # ==================================================
    # Buoyancy
    # ==================================================
    b_td = -g * (rho_td - rho0) / rho0
    b_td_prev = -g * (rho_td_prev - rho0) / rho0

    b = -g * (rho - rho0) / rho0
    bs = b.isel(k=0)

    # ==================================================
    # (1) TRUE LHS
    # ∫_{-H}^0 ∂_t b dz
    # ==================================================
    db_dt = (b_td - b_td_prev) / dt

    # CORRECT mixed-layer mask
    # depth: (k), Hml: (j,i) → broadcasts correctly
    # ml_mask = (depth >= -Hml) & (Hml >= min_H)

    # LHS_true = (db_dt * dz3d).where(ml_mask).sum("k")

    ml_mask = depth >= -Hml

    LHS_true = (db_dt * dz3d).where(ml_mask).sum("k").where(Hml >= min_H)

    # ==================================================
    # (2) ESTIMATED LHS
    # ==================================================
    bs_td = b_td.isel(k=0)
    bs_td_prev = b_td_prev.isel(k=0)

    dbs_dt = (bs_td - bs_td_prev) / dt
    dH_dt = (Hml_td - Hml_td_prev) / dt

    LHS_bs = (
        (2.0 * bs - 1.5 * g * delta_rho / rho0) * dH_dt
        + Hml * dbs_dt
    ).where(Hml >= min_H)

    # ==================================================
    # Save output
    # ==================================================
    ds_out = xr.Dataset(
        data_vars=dict(
            LHS_true=LHS_true,
            LHS_bs=LHS_bs,
        ),
        coords=dict(
            i=ds_rho["i"],
            j=ds_rho["j"],
            lat=lat,
            lon=lon,
        ),
        attrs=dict(
            description="Mixed-layer integrated buoyancy tendency",
            convention="depth negative downward",
        ),
    )

    ds_out.to_netcdf(out_file)
    print(f"Saved → {out_file}")

# ======================================================
# Cleanup
# ======================================================
client.close()
cluster.close()
print("Done.")
