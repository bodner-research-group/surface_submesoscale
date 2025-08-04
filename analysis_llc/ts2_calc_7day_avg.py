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
##### Step 2: compute 7-day averages of potential density, alpha, beta, Hml, save as .nc files

# ========== Imports ==========
import xarray as xr
import numpy as np
import gsw
import os
from glob import glob
from dask.distributed import Client, LocalCluster

# ========== Dask cluster setup ==========
cluster = LocalCluster(
    n_workers=64,             # 1 worker per CPU core
    threads_per_worker=1,     # avoid Python GIL
    memory_limit="5GB"        # total = 320 GB < 386 GB limit
)
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Paths ==========
input_dir = "/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/icelandic_basin"
output_dir = os.path.join(input_dir, "rho_7d_results")
os.makedirs(output_dir, exist_ok=True)

# ========== Open LLC4320 Dataset ==========
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

face = 2
i = slice(527, 1007)
j = slice(2960, 3441)
lat = ds1.YC.isel(face=face,i=1,j=j)
lon = ds1.XC.isel(face=face,i=i,j=1)
print(lat.chunks)
print(lon.chunks)

lon = lon.chunk({'i': -1})  # Re-chunk to include all data points
print(lon.chunks)

depth = ds1.Z

# ========== Broadcast lon, lat, depth ==========
lon_b, lat_b = xr.broadcast(lon, lat)
depth3d, _, _ = xr.broadcast(depth, lon_b, lat_b)

# ========== Hml computation function ==========
def compute_Hml(rho_profile, depth_profile, threshold=0.03):
    rho_10m = rho_profile[6]  # density at ~10m depth
    mask = rho_profile <= rho_10m + threshold
    if not np.any(mask):
        return 0.0
    return float(depth_profile[mask].max())

# ========== Main loop ==========
tt_files = sorted(glob(os.path.join(input_dir, "tt_12h_*.nc")))

for tt_file in tt_files:
    date_tag = os.path.basename(tt_file).replace("tt_12h_", "").replace(".nc", "")
    ss_file = tt_file.replace("tt", "ss")
    
    print(f"\nProcessing {date_tag}...")

    # ========== Read T, S ==========
    ds_tt = xr.open_dataset(tt_file)
    ds_ss = xr.open_dataset(ss_file)

    tt = ds_tt["Theta"]  # (time, k, j, i)
    ss = ds_ss["Salt"]

    # ========== Calculate 7-day averages ==========
    T_7d = tt.mean("time")
    S_7d = ss.mean("time")

    # ========== Calculate SA and CT ==========
    SA = xr.apply_ufunc(
        gsw.SA_from_SP, S_7d, depth3d, lon_b, lat_b,
        input_core_dims=[["k", "j", "i"], ["k", "j", "i"], ["j", "i"], ["j", "i"]],
        output_core_dims=[["k", "j", "i"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    CT = xr.apply_ufunc(
        gsw.CT_from_pt, SA, T_7d,
        input_core_dims=[["k", "j", "i"], ["k", "j", "i"]],
        output_core_dims=[["k", "j", "i"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # ========== Calculate rho, alpha, beta ==========
    p_ref = 0
    rho = xr.apply_ufunc(
        gsw.rho, SA, CT, p_ref,
        input_core_dims=[["k", "j", "i"], ["k", "j", "i"], []],
        output_core_dims=[["k", "j", "i"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    alpha = xr.apply_ufunc(
        gsw.alpha, SA, CT, depth3d,
        input_core_dims=[["k", "j", "i"]]*3,
        output_core_dims=[["k", "j", "i"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    beta = xr.apply_ufunc(
        gsw.beta, SA, CT, depth3d,
        input_core_dims=[["k", "j", "i"]]*3,
        output_core_dims=[["k", "j", "i"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # ========== Calculate Hml ==========
    Hml = xr.apply_ufunc(
        compute_Hml,
        rho,
        depth,
        input_core_dims=[["k"], ["k"]],
        output_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    # ========== Save results ==========
    out_ds = xr.Dataset({
        "rho_7d": rho,
        "alpha_7d": alpha,
        "beta_7d": beta,
        "Hml_7d": Hml
    })

    out_path = os.path.join(output_dir, f"rho_Hml_7d_{date_tag}.nc")
    out_ds.to_netcdf(out_path)

    print(f"Saved: {out_path}")

    # ========== Cleanup ==========
    ds_tt.close()
    ds_ss.close()
    out_ds.close()