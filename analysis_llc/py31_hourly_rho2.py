

##### Compute daily averaged potential density, alpha, beta, Hml, save as .nc files
# ========== Imports ==========
import xarray as xr
import numpy as np
import gsw
import os
import gc
from glob import glob
from dask.distributed import Client, LocalCluster

# from set_constant import domain_name, face, i, j
# ========== Domain ==========
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)   # icelandic_basin -- larger domain
j = slice(2960, 3441)  # icelandic_basin -- larger domain

# ========== Time settings ==========
ndays = 30
start_hours = (49 + 61) * 24
end_hours = start_hours + 24 * ndays
time = slice(start_hours,end_hours)

def main():
    # ========== Dask cluster setup ==========
    cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    # ========== Paths ==========
    output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/wb_hourly"
    os.makedirs(output_dir, exist_ok=True)

    # ========== Open LLC4320 Dataset ==========
    ds1 = xr.open_zarr("/orcd/data/abodner/003/LLC4320/LLC4320", consolidated=False)

    lat = ds1.YC.isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})
    lon = ds1.XC.isel(face=face, i=i, j=j).chunk({"j": -1, "i": -1})
    depth = ds1.Z
    depth3d, _, _ = xr.broadcast(np.abs(depth), lon, lat)

    tt_all = ds1.Theta.isel(face=face, i=i, j=j, time=time).chunk({"time": 24, "j": -1, "i": -1})
    ss_all = ds1.Salt.isel(face=face, i=i, j=j, time=time).chunk({"time": 24, "j": -1, "i": -1})

    num_times = tt_all.time.size


    # ========== Hml computation function ==========
    def compute_Hml_10mRef(rho_profile, depth_profile, threshold=0.03):
        rho_10m = rho_profile[6]  # density at ~ -9.66m depth
        mask = rho_profile > rho_10m + threshold
        if not np.any(mask):
            return 0.0
        return float(depth_profile[mask].max())

    def compute_Hml_SurfRef(rho_profile, depth_profile, threshold=0.03):
        rho_10m = rho_profile[0]  # density at ~ -0.5m depth
        mask = rho_profile > rho_10m + threshold
        if not np.any(mask):
            return 0.0
        return float(depth_profile[mask].max())


    # ========== Main Loop ==========
    # for t in range(num_times):
    for t in range(num_times-1, -1, -1):

        tt = tt_all.isel(time=t)
        ss = ss_all.isel(time=t)

        hour_tag = str(tt_all.time[t].dt.strftime("%Y-%m-%d-%H").values)
        print(f"\nProcessing hourly output for: {hour_tag}")

        out_path = os.path.join(output_dir, f"rho_Hml_{hour_tag}.nc")
        if os.path.exists(out_path):
            print(f"⏭️  Skipping {hour_tag}, output already exists.")
            continue

        # ========== Calculate SA and CT ==========
        SA = xr.apply_ufunc(
            gsw.SA_from_SP, ss, depth3d, lon, lat,
            input_core_dims=[["k", "j", "i"], ["k", "j", "i"], ["j", "i"], ["j", "i"]],
            output_core_dims=[["k", "j", "i"]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        CT = xr.apply_ufunc(
            gsw.CT_from_pt, SA, tt,
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
        Hml_10mRef = xr.apply_ufunc(
            compute_Hml_10mRef,
            rho,
            depth,
            input_core_dims=[["k"], ["k"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[float],
        )

        # ========== Compute Mixed Layer Depth ==========
        Hml_SurfRef = xr.apply_ufunc(
            compute_Hml_SurfRef,
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
            "rho": rho,
            "Hml_10mRef": Hml_10mRef,
            "Hml_SurfRef": Hml_SurfRef
        })

        out_ds.to_netcdf(out_path)
        print(f"Saved: {out_path}")

        # Cleanup
        del tt, ss, SA, CT, rho, Hml_10mRef, Hml_SurfRef, out_ds
        gc.collect()

    # ========== Cleanup ==========
    ds1.close()
    client.close()
    cluster.close()

# ========== Entry Point ==========
if __name__ == "__main__":
    main()
