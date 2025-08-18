##### Compute 7-day averages of potential density, alpha, beta, Hml, save as .nc files

# ========== Imports ==========
import xarray as xr
import numpy as np
import gsw
import os
from glob import glob
from dask.distributed import Client, LocalCluster

from set_constant import domain_name, face, i, j, start_hours, end_hours, step_hours

# ========== Hml computation function ==========
def compute_Hml(rho_profile, depth_profile, threshold=0.03):
    rho_10m = rho_profile[6]  # density at ~10m depth
    mask = rho_profile > rho_10m + threshold
    if not np.any(mask):
        return 0.0
    return float(depth_profile[mask].max())


def main():
    # ========== Dask cluster setup ==========
    cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    # ========== Paths ==========
    input_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/TSW_24h_avg"
    output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/rho_Hml_TS_weekly"
    os.makedirs(output_dir, exist_ok=True)

    # ========== Open LLC4320 Dataset ==========
    ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

    lat = ds1.YC.isel(face=face, i=1, j=j)
    lon = ds1.XC.isel(face=face, i=i, j=1)
    lon = lon.chunk({'i': -1}) # Re-chunk to include all data points
    lat = lat.chunk({'j': -1}) # Re-chunk to include all data points
    depth = ds1.Z

    # ========== Broadcast lon, lat, depth ==========
    lon_b, lat_b = xr.broadcast(lon, lat)
    depth3d, _, _ = xr.broadcast(depth, lon_b, lat_b)

    # ========== Main loop ==========
    tt_files = sorted(glob(os.path.join(input_dir, "tt_24h_*.nc")))

    for tt_file in tt_files:

        date_tag = os.path.basename(tt_file).replace("tt_24h_", "").replace(".nc", "")
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

        # Compute mixed layer depth
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

        # ========== Cleanup ==========
        ds_tt.close()
        ds_ss.close()
        out_ds.close()

    client.close()
    cluster.close()


# ========== Entry Point ==========
if __name__ == "__main__":
    main()
