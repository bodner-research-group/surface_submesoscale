import os
import numpy as np
import xarray as xr
import gsw
from dask.distributed import Client, LocalCluster
from glob import glob
from tqdm import tqdm

from set_constant import domain_name, face, i, j

# =======================
# Setup Dask cluster
# =======================
cluster = LocalCluster(n_workers=32, threads_per_worker=1, memory_limit="11GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Constants ==========
g = 9.81
p0 = 0  # surface pressure offset in dbar

# ========== Input Paths ==========
input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_7d_rolling_mean"
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_insitu_hydrostatic_pressure_7d_rolling_mean"
os.makedirs(output_dir, exist_ok=True)

# ========== Load static variables from LLC dataset ==========
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)
depth = ds1.Z
drF = ds1.drF
lon = ds1['XC'].isel(face=face, i=i, j=j)
lat = ds1['YC'].isel(face=face, i=i, j=j)

# Broadcast depth and drF to 3D
drF3d, _, _ = xr.broadcast(drF, lon, lat)
depth3d, _, _ = xr.broadcast(depth, lon, lat)

# ========= Define Column Processing Function ==========
def process_column(salt_col, theta_col, depth_col, drF_col, lon, lat):
    Nr = len(salt_col)
    
    # Prepare empty arrays
    SA_col = np.zeros(Nr)
    CT_col = np.zeros(Nr)
    rho_col = np.zeros(Nr)
    pres_col = np.zeros(Nr)
    
    ### Loop over vertical levels
    ### LLC4320 uses cell-centered approach: https://mitgcm.readthedocs.io/en/latest/algorithm/vert-grid.html
    for k in range(Nr):
        z = np.abs(depth_col[k])
        sp = salt_col[k]
        th = theta_col[k]

        if k == 0:
            #### Use depth to approximate pressure at level 0
            pres_k_estimated = z
            SA_k = gsw.SA_from_SP(sp, pres_k_estimated, lon, lat)
            CT_k = gsw.CT_from_pt(SA_k, th)
            rho_k = gsw.rho(SA_k, CT_k, pres_k_estimated)
            pres_k = p0 + g * rho_k * 0.5 * drF_col[k] / 1e4
        else:
            rho_prev = rho_col[k - 1]
            pres_prev = pres_col[k - 1]

            #### Use rho_prev to approximate rho_k when estimating pressure at level k
            pres_k_estimated = pres_prev + g * (rho_prev * 0.5 * drF_col[k - 1] + rho_prev * 0.5 * drF_col[k]) / 1e4
            SA_k = gsw.SA_from_SP(sp, pres_k_estimated, lon, lat)
            CT_k = gsw.CT_from_pt(SA_k, th)
            rho_k = gsw.rho(SA_k, CT_k, pres_k_estimated)
            pres_k = pres_prev + g * (rho_prev * 0.5 * drF_col[k - 1] + rho_k * 0.5 * drF_col[k]) / 1e4

        # Assign back to xarray objects
        SA_col[k] = SA_k
        CT_col[k] = CT_k
        rho_col[k] = rho_k
        pres_col[k] = pres_k

    SA_col = np.array(SA_col)
    CT_col = np.array(CT_col)
    rho_col = np.array(rho_col)
    pres_col = np.array(pres_col)

    return SA_col, CT_col, rho_col, pres_col

# ========= Batch Process All Files ==========
input_files = sorted(glob(os.path.join(input_dir, "rho_Hml_TS_7d_*.nc")))

for input_file in tqdm(input_files, desc="Processing time steps"):
    # input_file = os.path.join(input_dir, "rho_Hml_TS_7d_20120421.nc")
    date_tag = os.path.basename(input_file).split("_")[-1].replace(".nc", "")
    output_file = os.path.join(output_dir, f"rho_insitu_pres_hydro_{date_tag}.nc")

    if os.path.exists(output_file):
        print(f"Already processed: {output_file}")
        continue

    ds = xr.open_dataset(input_file, chunks={"i": 50, "j": 50, "k": -1})
    salt = ds["S_7d"]
    theta = ds["T_7d"]

    # Broadcast static fields
    drF3d, _, _ = xr.broadcast(drF, salt, salt)
    depth3d, _, _ = xr.broadcast(depth, salt, salt)

    # Apply processing function over (j, i) columns
    results = xr.apply_ufunc(
        process_column,
        salt,
        theta,
        depth3d,
        drF3d,
        lon,
        lat,
        input_core_dims=[["k"], ["k"], ["k"], ["k"], [], []],
        output_core_dims=[["k"], ["k"], ["k"], ["k"]],
        vectorize=True,
        dask="parallelized",
        dask_gufunc_kwargs={"output_sizes": {"k": depth.size}}, 
        output_dtypes=[salt.dtype, theta.dtype, salt.dtype, salt.dtype],
    )

    SA, CT, rho_insitu, pres_hydro = results  
    client.cancel(results)

    # Save output
    ds_out = xr.Dataset(
        {
            "SA": SA,
            "CT": CT,
            "rho_insitu": rho_insitu,
            "pres_hydro": pres_hydro
        },
        coords=ds.coords,
        attrs={"description": "Dask-parallelized vertical integration in-situ density and hydrostatic pressure"}
    )

    ds_out.to_netcdf(output_file)
    print(f"Saved: {output_file}")
