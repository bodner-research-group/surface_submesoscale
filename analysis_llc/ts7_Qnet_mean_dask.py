##### Compute and plot daily and weekly-smoothed surface net heat flux, net fresh water flux, and net buoyancy flux

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import gsw
import dask
import pandas as pd
from timezonefinder import TimezoneFinder

from dask.distributed import Client, LocalCluster
from dask import compute

from set_constant import domain_name, face, i, j, start_hours, end_hours, step_hours

def main():
    # ========== Dask cluster setup ==========
    cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
    client = Client(cluster)
    print("Dask dashboard:", client.dashboard_link)

    try:
        # ========== Load dataset ==========
        ds1 = xr.open_zarr('/orcd/data/abodner/003/LLC4320/LLC4320', consolidated=False)

        # ========== Paths ==========
        figdir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/figs/{domain_name}"
        output_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}"

        # ========== Plot font ==========
        plt.rcParams.update({'font.size': 16})

        # ========== Coordinates for center location (optional) ==========
        lat = ds1.YC.isel(face=face, i=1, j=j)
        lon = ds1.XC.isel(face=face, i=i, j=1)
        lat_c = float(lat.mean().values)
        lon_c = float(lon.mean().values)

        # Local timezone (not used, but included for completeness)
        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lng=lon_c, lat=lat_c)

        # ========== Time range ==========
        nday_avg = 364
        start_hours = 49 * 24
        end_hours = start_hours + 24 * nday_avg
        time_avg = slice(start_hours, end_hours, 1)

        # ========== Load and chunk variables ==========
        oceQnet = ds1.oceQnet.isel(time=time_avg, face=face, i=i, j=j).chunk({'time': 24, 'j': -1, 'i': -1})
        oceFWflx = ds1.oceFWflx.isel(time=time_avg, face=face, i=i, j=j).chunk({'time': 24, 'j': -1, 'i': -1})
        print("Chunks:", oceQnet.chunks)

        oceQnet = oceQnet.load()
        oceFWflx = oceFWflx.load()

        # ========== Spatial mean ==========
        oceQnet_mean = oceQnet.mean(dim=["i", "j"])
        oceFWflx_mean = oceFWflx.mean(dim=["i", "j"])

        # ========== Assign datetime ==========
        time_vals = pd.to_datetime(ds1.time.isel(time=time_avg).values)
        oceQnet_mean["time"] = time_vals
        oceFWflx_mean["time"] = time_vals

        # ========== Daily averages ==========
        oceQnet_daily_vals = oceQnet_mean.coarsen(time=24, boundary='trim').mean()
        oceFWflx_daily_vals = oceFWflx_mean.coarsen(time=24, boundary='trim').mean()

        # ========== 7-day smoothing ==========
        oceQnet_smooth_vals = oceQnet_daily_vals.rolling(time=7, center=True).mean()
        oceFWflx_smooth_vals = oceFWflx_daily_vals.rolling(time=7, center=True).mean()

        # ========== Compute results ==========
        oceQnet_daily_vals, oceFWflx_daily_vals, oceQnet_smooth_vals, oceFWflx_smooth_vals = compute(
            oceQnet_daily_vals, oceFWflx_daily_vals, oceQnet_smooth_vals, oceFWflx_smooth_vals
        )

        # ========== Merge to dataset ==========
        ds_combined = xr.Dataset(
            {
                "qnet_daily_avg": oceQnet_daily_vals,
                "qnet_7day_smooth": oceQnet_smooth_vals,
                "fwflx_daily_avg": oceFWflx_daily_vals,
                "fwflx_7day_smooth": oceFWflx_smooth_vals,
            },
            attrs={
                "description": "Daily and 7-day smoothed surface heat and fresh water fluxes",
                "source": "Processed from LLC4320 model data",
            }
        )

        # ========== Save to NetCDF ==========
        output_nc_path = f"{output_dir}/qnet_fwflx_daily_7day.nc"
        ds_combined.to_netcdf(output_nc_path)
        print(f"Saved NetCDF to: {output_nc_path}")

        # ========== Load combined data for plotting ==========
        ds_combined = xr.open_dataset(output_nc_path)

        # ========== Constants for buoyancy flux ==========
        gravity = 9.81
        SA = 35.2
        CT = 6
        p = 0
        rho0 = 999.8  # kg/m^3
        Cp = 3975     # J/kg/K

        beta = gsw.beta(SA, CT, p)
        alpha = gsw.alpha(SA, CT, p)

        # ========== Compute buoyancy flux ==========
        Bflux_daily_avg = gravity * alpha * ds_combined.qnet_daily_avg / (rho0 * Cp) + \
                          gravity * beta * ds_combined.fwflx_daily_avg * SA / rho0

        Bflux_7day_smooth = gravity * alpha * ds_combined.qnet_7day_smooth / (rho0 * Cp) + \
                            gravity * beta * ds_combined.fwflx_7day_smooth * SA / rho0

        # ========== Plotting ==========
        fig, axs = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

        # Qnet
        axs[0].plot(ds_combined.time, ds_combined.qnet_daily_avg, label='Daily Avg Qnet', color='tab:red', alpha=0.4)
        axs[0].plot(ds_combined.time, ds_combined.qnet_7day_smooth, label='7-day Smoothed Qnet', color='tab:blue')
        axs[0].axhline(0, color='k', linestyle='--', linewidth=1)
        axs[0].set_ylabel('Qnet [W/m²]')
        axs[0].set_title('(a) Net Surface Heat Flux')
        axs[0].legend()
        axs[0].grid(True)

        # FWflx
        axs[1].plot(ds_combined.time, ds_combined.fwflx_daily_avg, label='Daily Avg FWflx', color='tab:red', alpha=0.4)
        axs[1].plot(ds_combined.time, ds_combined.fwflx_7day_smooth, label='7-day Smoothed FWflx', color='tab:blue')
        axs[1].axhline(0, color='k', linestyle='--', linewidth=1)
        axs[1].set_ylabel('FWflx [kg/m²/s]')
        axs[1].set_title('(b) Net Surface Fresh Water Flux')
        axs[1].legend()
        axs[1].grid(True)

        # Bflux
        axs[2].plot(ds_combined.time, Bflux_daily_avg, label='Daily Avg Bflx', color='tab:red', alpha=0.4)
        axs[2].plot(ds_combined.time, Bflux_7day_smooth, label='7-day Smoothed Bflx', color='tab:blue')
        axs[2].axhline(0, color='k', linestyle='--', linewidth=1)
        axs[2].set_ylabel('Bflx [m²/s³]')
        axs[2].set_title('(c) Net Surface Buoyancy Flux')
        axs[2].legend()
        axs[2].grid(True)

        plt.tight_layout()
        fig_path = f"{figdir}/combined_surface_fluxes.png"
        plt.savefig(fig_path, dpi=150)
        plt.close()
        print(f"Saved plot to: {fig_path}")

    finally:
        client.close()
        cluster.close()

# ========== Entry Point ==========
if __name__ == "__main__":
    main()
