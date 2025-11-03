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
# # ========== Domain ==========
# domain_name = "icelandic_basin"
# face = 2
# i = slice(527, 1007)   # icelandic_basin -- larger domain
# j = slice(2960, 3441)  # icelandic_basin -- larger domain

# ========== Hml computation function ==========
def compute_Hml(rho_profile, depth_profile, threshold=0.03):
    rho_10m = rho_profile[6]  # density at ~10m depth
    mask = rho_profile > rho_10m + threshold
    if not np.any(mask):
        return 0.0
    return float(depth_profile[mask].max())

def main():
    # ========== Dask cluster setup ==========
    cluster = LocalCluster(n_workers=32, threads_per_worker=1, memory_limit="11GB")
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    # ========== Paths ==========
    input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TSW_24h_avg"
    output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_daily_avg"
    os.makedirs(output_dir, exist_ok=True)

    # ========== Open LLC4320 Dataset ==========
    ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

    lat = ds1.YC.isel(face=face, i=1, j=j)
    lon = ds1.XC.isel(face=face, i=i, j=1)
    lon = lon.chunk({'i': -1})
    lat = lat.chunk({'j': -1})
    depth = ds1.Z

    # ========== Broadcast lon, lat, depth ==========
    lon_b, lat_b = xr.broadcast(lon, lat)
    depth3d, _, _ = xr.broadcast(np.abs(depth), lon_b, lat_b)

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
        T_daily = tt_all.isel(time=t)
        S_daily = ss_all.isel(time=t)

        date_tag = str(tt_all.time[t].dt.strftime("%Y%m%d").values)
        print(f"\nProcessing daily mean for: {date_tag}")

        # ========== Calculate SA and CT ==========
        SA = xr.apply_ufunc(
            gsw.SA_from_SP, S_daily, depth3d, lon_b, lat_b,
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

        alpha = xr.apply_ufunc(
            gsw.alpha, SA, CT, depth3d,
            input_core_dims=[["k", "j", "i"]] * 3,
            output_core_dims=[["k", "j", "i"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        beta = xr.apply_ufunc(
            gsw.beta, SA, CT, depth3d,
            input_core_dims=[["k", "j", "i"]] * 3,
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
            "T_daily": T_daily,
            "S_daily": S_daily,
            "rho_daily": rho,
            "alpha_daily": alpha,
            "beta_daily": beta,
            "Hml_daily": Hml
        })

        out_path = os.path.join(output_dir, f"rho_Hml_TS_daily_{date_tag}.nc")
        out_ds.to_netcdf(out_path)
        print(f"Saved: {out_path}")

        # Cleanup
        del T_daily, S_daily, SA, CT, rho, alpha, beta, Hml, out_ds
        gc.collect()

    # ========== Cleanup ==========
    ds_tt_all.close()
    ds_ss_all.close()
    client.close()
    cluster.close()

# ========== Entry Point ==========
if __name__ == "__main__":
    main()
