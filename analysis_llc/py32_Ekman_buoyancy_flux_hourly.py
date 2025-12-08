"""
Compute Ekman Buoyancy Flux (B_Ek) from LLC4320 hourly surf data
Memory-safe: process one day at a time
Output: 1 NetCDF per day
"""

import os
import numpy as np
import xarray as xr
import gsw
from dask.distributed import Client, LocalCluster
from xgcm import Grid

# ==================================================
# DASK CONFIG
# ==================================================
cluster = LocalCluster(n_workers=64, threads_per_worker=1, memory_limit="5.5GB")
client = Client(cluster)
print("Dask dashboard:", client.dashboard_link)

# ==================================================
# Domain settings
# ==================================================
from set_constant import domain_name, face, i, j, start_hours, end_hours

# ==================================================
# Parameters
# ==================================================
rho0 = 1027.5
gravity = 9.81

# ==================================================
# Paths
# ==================================================
in_zarr = "/orcd/data/abodner/003/LLC4320/LLC4320"
out_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux"
os.makedirs(out_dir, exist_ok=True)

# ==================================================
# Load grid (static)
# ==================================================
ds = xr.open_zarr(in_zarr, consolidated=False)

lon = ds.XC.isel(face=face, i=i, j=j)
lat = ds.YC.isel(face=face, i=i, j=j)

Coriolis = 4 * np.pi / 86164 * np.sin(lat * np.pi / 180)

# Setup xgcm grid
ds_grid = ds.isel(face=face, i=i, j=j, i_g=i, j_g=j, k=0, k_p1=0, k_u=0).isel(time=0, drop=True)

coords = {
    "X": {"center": "i", "left": "i_g"},
    "Y": {"center": "j", "left": "j_g"},
}
metrics = {
    ("X",): ["dxC", "dxG"],
    ("Y",): ["dyC", "dyG"],
}
grid = Grid(ds_grid, coords=coords, metrics=metrics, periodic=False)

# ==================================================
# Daily loop
# ==================================================
hours_per_day = 24
for hour0 in range(start_hours, end_hours, hours_per_day):

    hour1 = min(hour0 + hours_per_day, end_hours)
    print(f"\n===== PROCESSING HOURS {hour0} → {hour1} =====")

    # --------------------------------------------------
    # Subset 24 hours
    # --------------------------------------------------
    ds_day = ds.isel(face=face, i=i, j=j, time=slice(hour0, hour1))

    salt = ds_day.Salt.isel(k=0).chunk({"time": 24})
    theta = ds_day.Theta.isel(k=0).chunk({"time": 24})

    taux = ds_day.oceTAUX.isel(j=j, i_g=i).chunk({"time": 24})
    tauy = ds_day.oceTAUY.isel(i=i, j_g=j).chunk({"time": 24})

    # ==================================================
    # Compute SA, CT, rho
    # ==================================================
    SA = xr.apply_ufunc(
        gsw.SA_from_SP,
        salt, 0, lon, lat,
        input_core_dims=[["j","i"], [], ["j","i"], ["j","i"]],
        output_core_dims=[["j","i"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float32],
    )

    CT = xr.apply_ufunc(
        gsw.CT_from_pt,
        SA, theta,
        input_core_dims=[["j","i"], ["j","i"]],
        output_core_dims=[["j","i"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float32],
    )

    rho = xr.apply_ufunc(
        gsw.rho,
        SA, CT, 0,
        input_core_dims=[["j","i"], ["j","i"], []],
        output_core_dims=[["j","i"]],
        dask="parallelized",
        vectorize=True,
        output_dtypes=[np.float32],
    )

    buoy = -gravity * (rho - rho0) / rho0

    # ==================================================
    # Gradients
    # ==================================================
    dbdx = grid.interp(grid.derivative(buoy, axis="X"), axis="X", to="center")
    dbdy = grid.interp(grid.derivative(buoy, axis="Y"), axis="Y", to="center")

    # ==================================================
    # Wind stress to C-grid
    # ==================================================
    taux_c = grid.interp(taux, axis="X", to="center")
    tauy_c = grid.interp(tauy, axis="Y", to="center")

    # ==================================================
    # Ekman buoyancy flux
    # ==================================================
    B_Ek = (tauy_c * dbdx - taux_c * dbdy) / (rho0 * Coriolis)
    B_Ek = B_Ek.rename("B_Ek")
    B_Ek.attrs["units"] = "m^2/s^3"

    # ==================================================
    # Compute + Save
    # ==================================================
    print("Computing + writing daily file...")

    # out_nc = f"{out_dir}/B_Ek_day_{hour0//24:03d}.nc"

    # Get datetime of first timestep in this daily chunk
    t0 = ds_day.time.isel(time=0).load().values
    date_str = np.datetime_as_string(t0, unit="D")  # returns '2011-01-01'

    # Build filename: B_Ek_20110101.nc
    out_nc = f"{out_dir}/B_Ek_{date_str.replace('-', '')}.nc"

    # Compute in Dask, write to disk
    B_Ek.compute().to_netcdf(out_nc)

    print(f"Saved: {out_nc}")

print("ALL DONE.")












##### ------------------------------------------------------------
##### SCRIPT 2 — COMPUTE DOMAIN-MEAN TIMESERIES FROM DAILY FILES
##### ------------------------------------------------------------


import xarray as xr
import numpy as np
import os

from set_constant import domain_name

in_dir  = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux_daily"
out_nc  = f"{in_dir}/B_Ek_domain_mean_timeseries.nc"

files = sorted([f for f in os.listdir(in_dir) if f.startswith("B_Ek_day_")])

means = []
times = []

for f in files:
    ds = xr.open_dataset(os.path.join(in_dir, f))
    B = ds.B_Ek

    # remove boundary
    Bsub = B.isel(i=slice(2,-2), j=slice(2,-2))
    Bmean = Bsub.mean(["i","j"])

    means.append(Bmean.values)
    times.append(Bmean.time.values)

Bmean_ts = xr.DataArray(
    data=np.concatenate(means),
    coords={"time": np.concatenate(times)},
    dims="time",
    name="B_Ek_mean",
)
Bmean_ts.attrs["units"] = "m^2/s^3"
Bmean_ts.to_netcdf(out_nc)

print("Saved:", out_nc)










##### ------------------------------------------------------------
##### SCRIPT 3 — PLOT FULL TIMESERIES + ROLLING MEAN
##### ------------------------------------------------------------

import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

from set_constant import domain_name

in_dir = f"/orcd/data/abodner/002/ysi/surface_submesoscale/analysis_llc/data/{domain_name}/Ekman_buoyancy_flux_daily"
ts_file = f"{in_dir}/B_Ek_domain_mean_timeseries.nc"

ds = xr.open_dataset(ts_file)
B = ds.B_Ek_mean

# 7-day rolling
roll = B.rolling(time=24*7, center=True).mean()

plt.figure(figsize=(14,5))
plt.plot(B.time, B, label="Domain mean hourly")
plt.plot(B.time, roll, label="7-day rolling mean", lw=2)
plt.legend()
plt.grid()
plt.title("Domain-mean Ekman Buoyancy Flux")
plt.xlabel("Time")
plt.ylabel("B_Ek (m²/s³)")
plt.tight_layout()

plt.savefig(f"{in_dir}/B_Ek_timeseries.png", dpi=150)
print("Saved figure.")
