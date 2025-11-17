##### Compute 7-day rolling mean of potential density, alpha, beta, Hml, save as .nc files
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
    output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_7d_rolling_mean"
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

    ds_tt_all = xr.open_mfdataset(tt_files, concat_dim="time", combine="nested", chunks={"time": 7})
    ds_ss_all = xr.open_mfdataset(ss_files, concat_dim="time", combine="nested", chunks={"time": 7})

    tt_all = ds_tt_all["Theta"]  # (time, k, j, i)
    ss_all = ds_ss_all["Salt"]

    # ========== Rolling mean ==========
    T_rolling = tt_all.rolling(time=7, center=True).mean()
    S_rolling = ss_all.rolling(time=7, center=True).mean()

    num_times = T_rolling.time.size

    # ========== Loop over valid rolling steps ==========
    for t in range(num_times):
        T_7d = T_rolling.isel(time=t)
        S_7d = S_rolling.isel(time=t)

        # Skip NaN edges (first 3 and last 3 days)
        if np.isnan(T_7d.isel(k=0, j=0, i=0)).all():
            continue

        date_tag = str(T_rolling.time[t].dt.strftime("%Y%m%d").values)

        print(f"\nProcessing rolling day: {date_tag}")

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
            "T_7d": T_7d,
            "S_7d": S_7d,
            "rho_7d": rho,
            "alpha_7d": alpha,
            "beta_7d": beta,
            "Hml_7d": Hml
        })

        out_path = os.path.join(output_dir, f"rho_Hml_TS_7d_{date_tag}.nc")
        out_ds.to_netcdf(out_path)
        print(f"Saved: {out_path}")

        # Manually clean up between iterations
        del T_7d, S_7d, SA, CT, rho, alpha, beta, Hml, out_ds
        gc.collect()
        

    # ========== Cleanup ==========
    ds_tt_all.close()
    ds_ss_all.close()
    client.close()
    cluster.close()

# ========== Entry Point ==========
if __name__ == "__main__":
    main()
