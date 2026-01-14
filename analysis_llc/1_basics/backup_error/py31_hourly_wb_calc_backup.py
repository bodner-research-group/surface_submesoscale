#!/usr/bin/env python

import xarray as xr
import numpy as np
import os
import gc
from glob import glob

from dask.distributed import Client, LocalCluster
import dask


# ============================================================
#                       DOMAIN
# ============================================================
domain_name = "icelandic_basin"
face = 2
i = slice(527, 1007)
j = slice(2960, 3441)


def main():

    # ============================================================
    #                       DASK SETUP
    # ============================================================
    # IMPORTANT:
    # On a single Slurm node, the most stable setup is:
    #   - ONE worker
    #   - MANY threads
    #
    # This avoids inter-process NetCDF / HDF5 crashes.
    cluster = LocalCluster(
        n_workers=1,
        threads_per_worker=64,
        memory_limit="360GB",
        dashboard_address=":8787",
    )
    client = Client(cluster)

    print("Dask dashboard:", client.dashboard_link)

    # Make Dask a bit more conservative
    dask.config.set({
        "distributed.worker.memory.target": 0.80,
        "distributed.worker.memory.spill": 0.85,
        "distributed.worker.memory.pause": 0.90,
        "distributed.worker.memory.terminate": 0.95,
    })

    # ============================================================
    #                       PATHS
    # ============================================================
    rho_dir = (
        f"/orcd/data/abodner/002/ysi/surface_submesoscale/"
        f"analysis_llc/data/{domain_name}/hourly_rho_Hml"
    )

    out_dir = (
        f"/orcd/data/abodner/002/ysi/surface_submesoscale/"
        f"analysis_llc/data/{domain_name}/hourly_wb_eddy_window26"
    )
    os.makedirs(out_dir, exist_ok=True)

    # ============================================================
    #                     LOAD MODEL GRID
    # ============================================================
    # Open once, reused for all files
    ds_grid = xr.open_zarr(
        "/orcd/data/abodner/003/LLC4320/LLC4320",
        consolidated=False,
    )

    lat = ds_grid.YC.isel(face=face, i=i, j=j)
    lon = ds_grid.XC.isel(face=face, i=i, j=j)

    depth = ds_grid.Z          # (k)
    depth_kp1 = ds_grid.Zp1    # (k+1)
    dz = ds_grid.drF           # (k)

    # Broadcast dz to 3D once
    dz3d, _, _ = xr.broadcast(dz, lon, lat)

    # ============================================================
    #                 HOURLY VERTICAL VELOCITY
    # ============================================================
    ndays = 366
    start_hours = 49 * 24
    end_hours = start_hours + 24 * ndays
    time = slice(start_hours, end_hours)

    W_all = (
        ds_grid.W
        .isel(face=face, i=i, j=j, time=time)
        .chunk({"time": 1, "k_p1": -1, "j": -1, "i": -1})
    )

    # ============================================================
    #                 MIXED-LAYER INTEGRAL
    # ============================================================
    def ml_integrated_profile(var, Hml, depth, dz, min_H):
        if Hml < min_H:
            return np.nan

        mask = depth >= -Hml
        if not np.any(mask):
            return np.nan

        num = np.sum(var[mask] * dz[mask])
        den = np.sum(dz[mask])

        if den == 0:
            return np.nan

        return num / den

    def ml_integral(var, Hml, depth, dz, min_H):
        return xr.apply_ufunc(
            ml_integrated_profile,
            var,
            Hml,
            depth,
            dz,
            input_core_dims=[["k"], [], ["k"], ["k"]],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            kwargs={"min_H": min_H},
            output_dtypes=[float],
        )

    # ============================================================
    #                       CONSTANTS
    # ============================================================
    gravity = 9.81
    rho0 = 1027.5
    min_H = 10.0
    window = 26  # 26/48 degree

    # ============================================================
    #                     FILE LIST
    # ============================================================
    rho_files = sorted(glob(os.path.join(rho_dir, "rho_Hml_*.nc")))

    # Adjust this slice as needed
    rho_files = rho_files[61 * 24 : 91 * 24]

    print(f"Number of files to process: {len(rho_files)}")

    # ============================================================
    #                     MAIN LOOP (SERIAL FILE IO)
    # ============================================================
    for f in rho_files:

        tag = os.path.basename(f).replace("rho_Hml_", "").replace(".nc", "")
        out_file = os.path.join(out_dir, f"wb_eddy_{tag}.nc")

        if os.path.exists(out_file):
            print(f"[SKIP] {tag}")
            continue

        print(f"[PROCESS] {tag}")

        # --------------------------------------------------------
        # Load rho / Hml
        # --------------------------------------------------------
        ds_rho = xr.open_dataset(f)

        rho = ds_rho["rho"].chunk({"j": -1, "i": -1})
        Hml = -ds_rho["Hml_SurfRef"].chunk({"j": -1, "i": -1})

        # --------------------------------------------------------
        # Match W time index
        # --------------------------------------------------------
        t_index = int(
            np.argmin(np.abs(W_all.time.values - rho.time.values))
        )

        w_kp1 = W_all.isel(time=t_index)

        # Interpolate W(k+1) -> W(k)
        w_k = (
            w_kp1.rename({"k_p1": "k"})
            .assign_coords(k=depth_kp1.values)
            .interp(k=depth)
            .fillna(0.0)
        )

        # --------------------------------------------------------
        # Buoyancy
        # --------------------------------------------------------
        b = -gravity * (rho - rho0) / rho0

        # --------------------------------------------------------
        # Coarse graining
        # --------------------------------------------------------
        Hml_cg = (
            Hml
            .coarsen(i=window, j=window, boundary="trim")
            .mean()
            .rename("Hml_mld")
        )

        b_cg = b.coarsen(i=window, j=window, boundary="trim").mean()
        w_cg = w_k.coarsen(i=window, j=window, boundary="trim").mean()
        dz_cg = dz3d.coarsen(i=window, j=window, boundary="trim").mean()

        wb_cg = (w_k * b).coarsen(i=window, j=window, boundary="trim").mean()

        # --------------------------------------------------------
        # Mixed-layer integrals
        # --------------------------------------------------------
        wb_avg = ml_integral(wb_cg, Hml_cg, depth, dz_cg, min_H)
        wb_fact = ml_integral(w_cg * b_cg, Hml_cg, depth, dz_cg, min_H)

        B_eddy = wb_avg - wb_fact

        # --------------------------------------------------------
        # FORCE COMPUTATION (CRITICAL)
        # --------------------------------------------------------
        wb_avg = wb_avg.load()
        wb_fact = wb_fact.load()
        B_eddy = B_eddy.load()
        Hml_cg = Hml_cg.load()

        # --------------------------------------------------------
        # Build output Dataset
        # --------------------------------------------------------
        ds_out = xr.Dataset(
            data_vars=dict(
                wb_avg=wb_avg,
                wb_fact=wb_fact,
                B_eddy=B_eddy,
                Hml_mld=Hml_cg,
            ),
            coords=dict(
                time=rho.time,
            ),
        )

        ds_out.load()

        # --------------------------------------------------------
        # WRITE NETCDF (SERIAL, SAFE)
        # --------------------------------------------------------
        ds_out.to_netcdf(out_file)
        print(f"[SAVED] {out_file}")

        # --------------------------------------------------------
        # CLEANUP
        # --------------------------------------------------------
        ds_rho.close()
        ds_out.close()
        del ds_out, rho, Hml, w_k, b
        gc.collect()

    # ============================================================
    #                     SHUTDOWN
    # ============================================================
    ds_grid.close()
    client.close()
    cluster.close()


# ============================================================
#                       ENTRY POINT
# ============================================================
if __name__ == "__main__":
    main()
