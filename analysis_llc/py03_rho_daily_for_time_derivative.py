##### Compute daily averaged potential density, alpha, beta, Hml, save as .nc files
# ========== Imports ==========
import xarray as xr
import numpy as np
import gsw
import os
import gc
from glob import glob
from dask.distributed import Client, LocalCluster

from set_constant import domain_name, face, i, j


# ========== Hml computation function ==========
def compute_Hml(rho_profile, depth_profile, threshold=0.03):
    # rho_10m = rho_profile[6]  # density at ~10m depth
    rho_surf = rho_profile[0]  # density at surface (z = -0.5 m)
    mask = rho_profile > rho_surf + threshold
    if not np.any(mask):
        return 0.0
    return float(depth_profile[mask].max())

# def main():
# ========== Dask cluster setup ==========
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ========== Paths ==========
# input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TSW_24h_avg"
# output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TS_24h_avg_for_time_derivative"
output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg_for_time_derivative"
os.makedirs(output_dir, exist_ok=True)

# ========== Open LLC4320 Dataset ==========
ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

lat = ds1.YC.isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})
lon = ds1.XC.isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})
depth = ds1.Z

# ========== Broadcast lon, lat, depth ==========
depth3d, _, _ = xr.broadcast(np.abs(depth), lon, lat)

# ========== Load All Files ==========
tt_files = sorted(glob(os.path.join(input_dir, "tt_24h_*.nc")))
ss_files = sorted(glob(os.path.join(input_dir, "ss_24h_*.nc")))

ds_tt_all = xr.open_mfdataset(tt_files, concat_dim="time", combine="nested", chunks={"time": 1})
ds_ss_all = xr.open_mfdataset(ss_files, concat_dim="time", combine="nested", chunks={"time": 1})

tt_all = ds_tt_all["Theta"]  # (time, k, j, i)
ss_all = ds_ss_all["Salt"]

num_times = tt_all.time.size

# ========== Loop over daily outputs ==========
for t in range(num_times):
# for t in range(num_times - 1, -1, -1):
    T_daily = tt_all.isel(time=t)
    S_daily = ss_all.isel(time=t)

    date_tag = str(tt_all.time[t].dt.strftime("%Y%m%d").values)
    print(f"\nProcessing daily mean for: {date_tag}")

    out_path = os.path.join(output_dir, f"rho_Hml_TS_daily_{date_tag}.nc")
    if os.path.exists(out_path):
        print(f"⏭️  Skipping {date_tag}, output already exists.")
        continue

    # ========== Calculate SA and CT ==========
    SA = xr.apply_ufunc(
        gsw.SA_from_SP, S_daily, depth3d, lon, lat,
        input_core_dims=[["k", "j", "i"], ["k", "j", "i"], ["j", "i"], ["j", "i"]],
        output_core_dims=[["k", "j", "i"]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=[float],
    )

    CT = xr.apply_ufunc(
        gsw.CT_from_pt, SA, T_daily,
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

    # ========== Compute Mixed Layer Depth ==========
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
        "rho_daily": rho,
        "Hml_daily": Hml
    })

    out_ds.to_netcdf(out_path)
    print(f"Saved: {out_path}")

    # Cleanup
    del T_daily, S_daily, SA, CT, rho, Hml, out_ds
    gc.collect()

# ========== Cleanup ==========
ds_tt_all.close()
ds_ss_all.close()
client.close()
cluster.close()

# # ========== Entry Point ==========
# if __name__ == "__main__":
#     main()
